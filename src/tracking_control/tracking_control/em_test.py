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

## Functions for quaternion and rotation matrix conversion
## The code is adapted from the general_robotics_toolbox package
## Code reference: https://github.com/rpiRobotics/rpi_general_robotics_toolbox_py
def hat(k):
    khat=np.zeros((3,3))
    khat[0,1]=-k[2]
    khat[0,2]=k[1]
    khat[1,0]=k[2]
    khat[1,2]=-k[0]
    khat[2,0]=-k[1]
    khat[2,1]=k[0]
    return khat

def q2R(q):
    I = np.identity(3)
    qhat = hat(q[1:4])
    qhat2 = qhat.dot(qhat)
    return I + 2*q[0]*qhat + 2*qhat2

def euler_from_quaternion(q):
    w=q[0]
    x=q[1]
    y=q[2]
    z=q[3]
    roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = math.asin(2 * (w * y - z * x))
    yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return [roll,pitch,yaw]

class TrackingNode(Node):
    def __init__(self):
        super().__init__('tracking_node')
        self.get_logger().info('Tracking Node Started')
        
        # Current object pose
        self.obs_pose = None
        self.goal_pose = None
        
        # ROS parameters
        self.declare_parameter('world_frame_id', 'odom')

        # Create a transform listener
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
        # Create publisher for the control command
        self.pub_control_cmd = self.create_publisher(Twist, '/track_cmd_vel', 10)

        # Only track the detected object color
        self.sub_detected_object_pose = self.create_subscription(
            PoseStamped,
            'detected_color_object_pose',
            self.detected_obs_pose_callback,
            10
        )

        # Create timer, running at 100Hz
        self.timer = self.create_timer(0.01, self.timer_update)
    
    def detected_obs_pose_callback(self, msg):
        odom_id = self.get_parameter('world_frame_id').get_parameter_value().string_value
        center_points = np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z])
        
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
        
        self.obs_pose = cp_world

    def get_current_poses(self):
        odom_id = self.get_parameter('world_frame_id').get_parameter_value().string_value
    
        try:
            # Get robot pose in the world (odom) frame
            transform = self.tf_buffer.lookup_transform(
                odom_id,
                'base_footprint',
                rclpy.time.Time()
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
    
            # self.obs_pose is already stored in odom/world frame
            # so first get vector from robot to object in world frame
            rel_world = self.obs_pose - robot_world_pos
    
            # Convert that world-frame vector into robot frame
            object_pose_robot = robot_world_R.T @ rel_world
    
        except TransformException as e:
            self.get_logger().error('Transform error: ' + str(e))
            return None
    
        return object_pose_robot
    
    def timer_update(self):
        if self.obs_pose is None:
            cmd_vel = Twist()
            cmd_vel.linear.x = 0.0
            cmd_vel.angular.z = 0.0
            self.pub_control_cmd.publish(cmd_vel)
            return
        
        current_obs_pose = self.get_current_poses()
        if current_obs_pose is None:
            return
        
        cmd_vel = self.controller(current_obs_pose)
        self.pub_control_cmd.publish(cmd_vel)
    
    def controller(self, obs_pose):
        cmd_vel = Twist()

        x = obs_pose[0]
        y = obs_pose[1]

        heading_error = math.atan2(y, x)
        distance = math.sqrt(x**2 + y**2)

        stop_distance = 0.4
        k_linear = 0.3
        k_angular = 1.0

        cmd_vel.angular.z = k_angular * heading_error

        if distance > stop_distance:
            cmd_vel.linear.x = k_linear * (distance - stop_distance)
        else:
            cmd_vel.linear.x = 0.0

        return cmd_vel

def main(args=None):
    rclpy.init(args=args)
    tracking_node = TrackingNode()
    rclpy.spin(tracking_node)
    tracking_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
