# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 17:51:58 2026

@author: Emerson
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, PoseStamped
from tf2_ros import TransformException, Buffer, TransformListener
import numpy as np
import math

"""Quaternion / rotation helpers"""
def hat(k):
    khat = np.zeros((3, 3))
    khat[0, 1] = -k[2]
    khat[0, 2] = k[1]
    khat[1, 0] = k[2]
    khat[1, 2] = -k[0]
    khat[2, 0] = -k[1]
    khat[2, 1] = k[0]
    return khat

def q2R(q):
    I = np.identity(3)
    qhat = hat(q[1:4])
    qhat2 = qhat.dot(qhat)
    return I + 2 * q[0] * qhat + 2 * qhat2


class TrackingNode(Node):
    def __init__(self):
        super().__init__('tracking_node')
        self.get_logger().info('Tracking Node Started')

        """Latest detected object pose in world frame"""
        self.obs_pose = None

        """Whether we currently consider the object visible"""
        self.object_visible = False

        """Low-pass filter for pose smoothing
        Smaller alpha = smoother, larger alpha = more responsive
        """
        self.filter_alpha = 0.25

        """How long we trust the last detection before saying object is lost"""
        self.target_timeout = 0.3
        self.last_seen_time = None

        """ROS parameters"""
        self.declare_parameter('world_frame_id', 'odom')

        """TF listener"""
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        """Velocity publisher"""
        self.pub_control_cmd = self.create_publisher(Twist, '/track_cmd_vel', 10)

        """Subscribe to detected object pose"""
        self.sub_detected_object_pose = self.create_subscription(
            PoseStamped,
            'detected_color_object_pose',
            self.detected_obs_pose_callback,
            10
        )

        """Update control at 100 Hz"""
        self.timer = self.create_timer(0.01, self.timer_update)

    def detected_obs_pose_callback(self, msg):
        """Convert detected object position into world frame and store it"""
        odom_id = self.get_parameter('world_frame_id').get_parameter_value().string_value
        center_points = np.array([
            msg.pose.position.x,
            msg.pose.position.y,
            msg.pose.position.z
        ])

        try:
            transform = self.tf_buffer.lookup_transform(
                odom_id,
                msg.header.frame_id,
                rclpy.time.Time(),
                rclpy.duration.Duration(seconds=0.1)
            )

            t_R = q2R(np.array([
                transform.transform.rotation.w,
                transform.transform.rotation.x,
                transform.transform.rotation.y,
                transform.transform.rotation.z
            ]))

            cp_world = t_R @ center_points + np.array([
                transform.transform.translation.x,
                transform.transform.translation.y,
                transform.transform.translation.z
            ])

        except TransformException as e:
            self.get_logger().error('Transform Error: {}'.format(e))
            return

        """Filter the pose so tracking is less jumpy"""
        if self.obs_pose is None:
            self.obs_pose = cp_world
        else:
            self.obs_pose = self.filter_alpha * cp_world + (1.0 - self.filter_alpha) * self.obs_pose

        """Mark object as visible and refresh last-seen time"""
        self.object_visible = True
        self.last_seen_time = self.get_clock().now()

    def object_recently_seen(self):
        """Return True if the object was seen recently enough to still track it"""
        if self.last_seen_time is None:
            return False

        dt = (self.get_clock().now() - self.last_seen_time).nanoseconds / 1e9
        return dt <= self.target_timeout

    def get_current_object_pose_in_robot_frame(self):
        """Convert saved world-frame object pose into robot frame"""
        odom_id = self.get_parameter('world_frame_id').get_parameter_value().string_value

        try:
            transform = self.tf_buffer.lookup_transform(
                'base_footprint',
                odom_id,
                rclpy.time.Time()
            )

            robot_world_x = transform.transform.translation.x
            robot_world_y = transform.transform.translation.y
            robot_world_z = transform.transform.translation.z

            robot_world_R = q2R(np.array([
                transform.transform.rotation.w,
                transform.transform.rotation.x,
                transform.transform.rotation.y,
                transform.transform.rotation.z
            ]))

            object_pose = robot_world_R @ self.obs_pose + np.array([
                robot_world_x,
                robot_world_y,
                robot_world_z
            ])

        except TransformException as e:
            self.get_logger().error('Transform error: ' + str(e))
            return None

        return object_pose

    def timer_update(self):
        """Main behavior logic:
        1. If object is not seen, spin in place to search.
        2. If object is seen, turn to center it while driving toward it.
        3. Stop forward motion once within 0.2 m.
        """
        cmd_vel = Twist()

        """If no recent detection, go into search mode"""
        if (self.obs_pose is None) or (not self.object_recently_seen()):
            self.object_visible = False
            cmd_vel = self.search_controller()
            self.pub_control_cmd.publish(cmd_vel)
            return

        """Object is visible, so track it"""
        current_obs_pose = self.get_current_object_pose_in_robot_frame()
        if current_obs_pose is None:
            return

        cmd_vel = self.track_controller(current_obs_pose)
        self.pub_control_cmd.publish(cmd_vel)

    def search_controller(self):
        """Spin in place quickly until the object is detected"""
        cmd_vel = Twist()
        cmd_vel.linear.x = 0.0
        cmd_vel.angular.z = 2.5
        return cmd_vel

    def track_controller(self, obs_pose):
        """Turn to center object while driving toward it"""
        cmd_vel = Twist()
    
        x = obs_pose[0]
        y = obs_pose[1]
    
        heading_error = math.atan2(y, x)
        distance = math.sqrt(x**2 + y**2)
        abs_error = abs(heading_error)
    
        """Distance rules"""
        stop_distance = 0.2
        close_distance = 0.3
    
        """Forward motion tuning"""
        k_linear = 0.45
        max_linear = 0.30
        min_linear = 0.08
    
        """
        When close to the object, do not keep spinning around trying to
        perfectly center it. Just stop forward motion at stop_distance,
        and strongly reduce turning once inside close_distance.
        """
        if distance <= stop_distance:
            cmd_vel.linear.x = 0.0
            cmd_vel.angular.z = 0.0
            return cmd_vel
    
        """Forward drive"""
        cmd_vel.linear.x = k_linear * (distance - stop_distance)
    
        if cmd_vel.linear.x > max_linear:
            cmd_vel.linear.x = max_linear
        elif cmd_vel.linear.x < min_linear:
            cmd_vel.linear.x = min_linear
    
        """Angular control"""

        """
        Farther away:
        - stronger turning when object is off-center
        - gentler turning when nearly centered
        """
        if abs_error > 0.5:
            k_angular = 1.2
            max_angular = 0.9
        elif abs_error > 0.2:
            k_angular = 0.7
            max_angular = 0.5
        else:
            k_angular = 0.35
            max_angular = 0.2

        cmd_vel.angular.z = k_angular * heading_error

        if cmd_vel.angular.z > max_angular:
            cmd_vel.angular.z = max_angular
        elif cmd_vel.angular.z < -max_angular:
            cmd_vel.angular.z = -max_angular

        return cmd_vel


def main(args=None):
    rclpy.init(args=args)
    tracking_node = TrackingNode()
    rclpy.spin(tracking_node)
    tracking_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
