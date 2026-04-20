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

def euler_from_quaternion(q):
    w = q[0]
    x = q[1]
    y = q[2]
    z = q[3]
    roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = math.asin(2 * (w * y - z * x))
    yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return [roll, pitch, yaw]

class TrackingNode(Node):
    def __init__(self):
        super().__init__('tracking_node')
        self.get_logger().info('Tracking Node Started')

        """Latest detected orange-object pose in world frame"""
        self.obs_pose = None

        """ROS parameters"""
        self.declare_parameter('world_frame_id', 'odom')

        """TF listener"""
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        """Velocity publisher"""
        self.pub_control_cmd = self.create_publisher(Twist, '/track_cmd_vel', 10)

        """Subscribe only to the orange object pose"""
        self.sub_detected_object_pose = self.create_subscription(
            PoseStamped,
            'detected_color_object_pose',
            self.detected_obs_pose_callback,
            10
        )

        """Update control at 100 Hz"""
        self.timer = self.create_timer(0.01, self.timer_update)

    def detected_obs_pose_callback(self, msg):
        """Convert detected orange blob position into world frame"""
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

        """Always use the latest orange blob that is published"""
        self.obs_pose = cp_world

    def get_current_object_pose_in_robot_frame(self):
        """Convert saved world-frame object pose into robot frame"""
        odom_id = self.get_parameter('world_frame_id').get_parameter_value().string_value

        try:
            transform = self.tf_buffer.lookup_transform(
                'base_footprint',
                odom_id,
                rclpy.time.Time()
            )

            """Robot pose in world coordinates"""
            robot_world_t = np.array([
                transform.transform.translation.x,
                transform.transform.translation.y,
                transform.transform.translation.z
            ])

            robot_world_R = q2R(np.array([
                transform.transform.rotation.w,
                transform.transform.rotation.x,
                transform.transform.rotation.y,
                transform.transform.rotation.z
            ]))

            """
            OLD CODE (kept for reference)
            object_pose = robot_world_R @ self.obs_pose + np.array([
                robot_world_x,
                robot_world_y,
                robot_world_z
            ])
            """

            """
            NEW CODE
            First subtract robot world position from object world position,
            then rotate that relative vector into robot frame.
            """
            object_pose = robot_world_R @ (self.obs_pose - robot_world_t)

        except TransformException as e:
            self.get_logger().error('Transform error: ' + str(e))
            return None

        return object_pose

    def timer_update(self):
        """If no orange object is detected, stop. Otherwise follow it."""
        cmd_vel = Twist()

        if self.obs_pose is None:
            cmd_vel.linear.x = 0.0
            cmd_vel.angular.z = 0.0
            self.pub_control_cmd.publish(cmd_vel)
            return

        current_obs_pose = self.get_current_object_pose_in_robot_frame()
        if current_obs_pose is None:
            return

        cmd_vel = self.controller(current_obs_pose)
        self.pub_control_cmd.publish(cmd_vel)

    def controller(self, obs_pose):
        """Simple proportional controller to face and approach orange object"""
        cmd_vel = Twist()

        x = obs_pose[0]
        y = obs_pose[1]

        heading_error = math.atan2(y, x)
        distance = math.sqrt(x**2 + y**2)

        stop_distance = 0.3
        k_linear = 0.4
        k_angular = 1.5

        """
        When the object gets close, heading can jump around a lot.
        So reduce turning gain nearby to make motion smoother.
        """
        if distance < 0.45:
            k_angular = 0.6

        cmd_vel.angular.z = k_angular * heading_error

        if distance > stop_distance:
            cmd_vel.linear.x = k_linear * (distance - stop_distance)
        else:
            """
            Once close enough, stop.
            This prevents the robot from doing weird last-second turns
            when the object is right near it.
            """
            cmd_vel.linear.x = 0.0
            cmd_vel.angular.z = 0.0

        return cmd_vel

def main(args=None):
    rclpy.init(args=args)
    tracking_node = TrackingNode()
    rclpy.spin(tracking_node)
    tracking_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
