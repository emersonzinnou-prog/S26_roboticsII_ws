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

        """Current detected object pose in odom/world frame"""
        self.obs_pose = None

        """ROS parameter"""
        self.declare_parameter('world_frame_id', 'odom')

        """TF"""
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        """Publisher"""
        self.pub_control_cmd = self.create_publisher(Twist, '/cmd_vel', 10)

        """Subscriber"""
        self.sub_detected_object_pose = self.create_subscription(
            PoseStamped,
            'detected_color_object_pose',
            self.detected_obs_pose_callback,
            10
        )

        """Main control timer"""
        self.timer = self.create_timer(0.05, self.timer_update)

        """Counters for throttled debug printing"""
        self.no_object_counter = 0
        self.loop_counter = 0
        self.detect_counter = 0

        self.get_logger().info('Subscribed to: detected_color_object_pose')
        self.get_logger().info('Publishing cmds to: /track_cmd_vel')

    def detected_obs_pose_callback(self, msg):
        odom_id = self.get_parameter('world_frame_id').get_parameter_value().string_value

        center_points = np.array([
            msg.pose.position.x,
            msg.pose.position.y,
            msg.pose.position.z
        ])

        self.detect_counter += 1

        try:
            """Transform detected object from sensor frame into odom/world frame"""
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
            self.get_logger().error(f'DETECTION TF FAILED: {e}')
            return

        self.obs_pose = cp_world

        self.get_logger().info(
            f'OBJECT DETECTED #{self.detect_counter} | '
            f'sensor_frame={msg.header.frame_id} | '
            f'raw=({msg.pose.position.x:.3f}, {msg.pose.position.y:.3f}, {msg.pose.position.z:.3f}) | '
            f'world=({self.obs_pose[0]:.3f}, {self.obs_pose[1]:.3f}, {self.obs_pose[2]:.3f})'
        )

    def get_current_poses(self):
        odom_id = self.get_parameter('world_frame_id').get_parameter_value().string_value

        try:
            """Get robot pose in world frame"""
            transform = self.tf_buffer.lookup_transform(
                odom_id,
                'base_footprint',
                rclpy.time.Time(),
                rclpy.duration.Duration(seconds=0.1)
            )

            robot_world_pos = np.array([
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
            self.obs_pose is already in odom/world frame.
            Get vector from robot to object in world frame,
            then rotate into robot frame.
            """
            rel_world = self.obs_pose - robot_world_pos
            object_pose_robot = robot_world_R.T @ rel_world

        except TransformException as e:
            self.get_logger().error(f'ROBOT TF FAILED: {e}')
            return None

        return object_pose_robot

    def timer_update(self):
        self.loop_counter += 1

        """No object detected yet"""
        if self.obs_pose is None:
            cmd_vel = Twist()
            cmd_vel.linear.x = 0.0
            cmd_vel.angular.z = 0.0
            self.pub_control_cmd.publish(cmd_vel)

            self.no_object_counter += 1
            if self.no_object_counter % 20 == 0:
                self.get_logger().info('NO OBJECT DETECTED -> publishing stop command')

            return

        """Object exists, now convert to robot frame"""
        current_obs_pose = self.get_current_poses()
        if current_obs_pose is None:
            cmd_vel = Twist()
            cmd_vel.linear.x = 0.0
            cmd_vel.angular.z = 0.0
            self.pub_control_cmd.publish(cmd_vel)
            self.get_logger().info('OBJECT EXISTS BUT TF FAILED -> publishing stop command')
            return

        """Compute and publish command"""
        cmd_vel = self.controller(current_obs_pose)
        self.pub_control_cmd.publish(cmd_vel)

        """Throttle logs a little so terminal is readable"""
        if self.loop_counter % 10 == 0:
            x = current_obs_pose[0]
            y = current_obs_pose[1]
            distance = math.sqrt(x**2 + y**2)
            heading_error = math.atan2(y, x)

            self.get_logger().info(
                f'OBJECT IN ROBOT FRAME | x={x:.3f}, y={y:.3f}, dist={distance:.3f}, heading={heading_error:.3f}'
            )
            self.get_logger().info(
                f'PUBLISH CMD | linear.x={cmd_vel.linear.x:.3f}, angular.z={cmd_vel.angular.z:.3f}'
            )

    def controller(self, obs_pose):
        cmd_vel = Twist()

        x = obs_pose[0]
        y = obs_pose[1]

        heading_error = math.atan2(y, x)
        distance = math.sqrt(x**2 + y**2)

        """
        Tune these first if needed
        """
        stop_distance = 0.15
        k_linear = 0.25
        k_angular = 1.2

        """
        Important debug logic:
        If x < 0, object is behind robot in robot frame.
        Turn in place to find it instead of trying to drive weirdly.
        """
        if x < 0.0:
            cmd_vel.linear.x = 0.0
            cmd_vel.angular.z = 0.6 if y >= 0.0 else -0.6
            self.get_logger().info('OBJECT BEHIND ROBOT -> turning in place')
            return cmd_vel

        """
        If object is far off-center, turn in place first.
        This avoids sideways/circling behavior.
        """
        if abs(heading_error) > 0.35:
            cmd_vel.linear.x = 0.0
            cmd_vel.angular.z = k_angular * heading_error
            self.get_logger().info('TURNING IN PLACE TOWARD OBJECT')
            return cmd_vel

        """
        If close enough, stop.
        """
        if distance <= stop_distance:
            cmd_vel.linear.x = 0.0
            cmd_vel.angular.z = 0.0
            self.get_logger().info('AT GOAL DISTANCE -> stopping')
            return cmd_vel

        """
        Otherwise move forward and make small heading correction.
        """
        cmd_vel.linear.x = k_linear * (distance - stop_distance)
        cmd_vel.angular.z = 0.8 * heading_error

        """
        Clamp speeds so logs/behavior are easier to interpret
        """
        if cmd_vel.linear.x > 0.25:
            cmd_vel.linear.x = 0.25
        if cmd_vel.angular.z > 1.0:
            cmd_vel.angular.z = 1.0
        if cmd_vel.angular.z < -1.0:
            cmd_vel.angular.z = -1.0

        self.get_logger().info('MOVING TOWARD OBJECT')
        return cmd_vel


def main(args=None):
    rclpy.init(args=args)
    tracking_node = TrackingNode()
    rclpy.spin(tracking_node)
    tracking_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
