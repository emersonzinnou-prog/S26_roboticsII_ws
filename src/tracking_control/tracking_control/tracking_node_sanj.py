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
    khat[0,1]=-k[2]; khat[0,2]=k[1]
    khat[1,0]=k[2];  khat[1,2]=-k[0]
    khat[2,0]=-k[1]; khat[2,1]=k[0]
    return khat

def q2R(q):
    I = np.identity(3)
    qhat = hat(q[1:4])
    qhat2 = qhat.dot(qhat)
    return I + 2*q[0]*qhat + 2*qhat2

def euler_from_quaternion(q):
    w,x,y,z = q[0],q[1],q[2],q[3]
    roll  = math.atan2(2*(w*x+y*z), 1-2*(x*x+y*y))
    pitch = math.asin(2*(w*y-z*x))
    yaw   = math.atan2(2*(w*z+x*y), 1-2*(y*y+z*z))
    return [roll, pitch, yaw]

class TrackingNode(Node):
    def __init__(self):
        super().__init__('tracking_node')
        self.get_logger().info('Tracking Node Started')

        self.obs_pose  = None
        self.goal_pose = None

        # Start pose recorded once on first valid goal detection (odom frame, x-y)
        self.start_pose = None

        # State machine: 'go_to_goal' -> 'return_to_start' -> stop
        self.state = 'go_to_goal'

        self.declare_parameter('world_frame_id', 'odom')

        self.tf_buffer   = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.pub_control_cmd       = self.create_publisher(Twist, '/track_cmd_vel', 10)
        self.sub_detected_goal_pose = self.create_subscription(PoseStamped, 'detected_color_object_pose', self.detected_obs_pose_callback,  10)
        self.sub_detected_obs_pose  = self.create_subscription(PoseStamped, 'detected_color_goal_pose',   self.detected_goal_pose_callback, 10)

        self.timer = self.create_timer(0.01, self.timer_update)

    # ------------------------------------------------------------------
    def detected_obs_pose_callback(self, msg):
        odom_id      = self.get_parameter('world_frame_id').get_parameter_value().string_value
        center_points = np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z])

        # TODO: Filtering
        # Reject detections that are too far away or too high (likely noise/background).
        if np.linalg.norm(center_points) > 3.0 or center_points[2] > 0.7:
            return

        try:
            transform = self.tf_buffer.lookup_transform(odom_id, msg.header.frame_id,
                                                        rclpy.time.Time(),
                                                        rclpy.duration.Duration(seconds=0.1))
            t_R = q2R(np.array([transform.transform.rotation.w,
                                 transform.transform.rotation.x,
                                 transform.transform.rotation.y,
                                 transform.transform.rotation.z]))
            cp_world = t_R @ center_points + np.array([transform.transform.translation.x,
                                                        transform.transform.translation.y,
                                                        transform.transform.translation.z])
        except TransformException as e:
            self.get_logger().error('Transform Error: {}'.format(e))
            return

        self.obs_pose = cp_world

    # ------------------------------------------------------------------
    def detected_goal_pose_callback(self, msg):
        odom_id       = self.get_parameter('world_frame_id').get_parameter_value().string_value
        center_points = np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z])

        # TODO: Filtering
        # Reject detections that are too far away or too high (likely noise/background).
        if np.linalg.norm(center_points) > 3.0 or center_points[2] > 0.7:
            return

        try:
            transform = self.tf_buffer.lookup_transform(odom_id, msg.header.frame_id,
                                                        rclpy.time.Time(),
                                                        rclpy.duration.Duration(seconds=0.1))
            t_R = q2R(np.array([transform.transform.rotation.w,
                                 transform.transform.rotation.x,
                                 transform.transform.rotation.y,
                                 transform.transform.rotation.z]))
            cp_world = t_R @ center_points + np.array([transform.transform.translation.x,
                                                        transform.transform.translation.y,
                                                        transform.transform.translation.z])
        except TransformException as e:
            self.get_logger().error('Transform Error: {}'.format(e))
            return

        self.goal_pose = cp_world

    # ------------------------------------------------------------------
    def get_current_poses(self):
        odom_id = self.get_parameter('world_frame_id').get_parameter_value().string_value
        try:
            transform = self.tf_buffer.lookup_transform('base_footprint', odom_id, rclpy.time.Time())
            rx = transform.transform.translation.x
            ry = transform.transform.translation.y
            rz = transform.transform.translation.z
            R  = q2R([transform.transform.rotation.w,
                      transform.transform.rotation.x,
                      transform.transform.rotation.y,
                      transform.transform.rotation.z])
            obstacle_pose = R @ self.obs_pose  + np.array([rx, ry, rz])
            goal_pose     = R @ self.goal_pose + np.array([rx, ry, rz])
        except TransformException as e:
            self.get_logger().error('Transform error: ' + str(e))
            return None, None
        return obstacle_pose, goal_pose

    def get_robot_pose_in_odom(self):
        """Returns (x, y, yaw) of base_footprint in the odom frame."""
        odom_id = self.get_parameter('world_frame_id').get_parameter_value().string_value
        try:
            transform = self.tf_buffer.lookup_transform(odom_id, 'base_footprint', rclpy.time.Time())
            x   = transform.transform.translation.x
            y   = transform.transform.translation.y
            yaw = euler_from_quaternion([transform.transform.rotation.w,
                                         transform.transform.rotation.x,
                                         transform.transform.rotation.y,
                                         transform.transform.rotation.z])[2]
            return x, y, yaw
        except TransformException as e:
            self.get_logger().error('get_robot_pose_in_odom error: ' + str(e))
            return None, None, None

    # ------------------------------------------------------------------
    def timer_update(self):
        ################### Write your code here ###################

        if self.goal_pose is None:
            cmd_vel = Twist()
            cmd_vel.linear.x  = 0.0
            cmd_vel.angular.z = 0.0
            self.pub_control_cmd.publish(cmd_vel)
            return

        # Record start pose once
        if self.start_pose is None:
            rx, ry, _ = self.get_robot_pose_in_odom()
            if rx is not None:
                self.start_pose = np.array([rx, ry])
                self.get_logger().info(f'Start pose recorded: {self.start_pose}')

        # Get poses in base_footprint frame
        current_obs_pose, current_goal_pose = self.get_current_poses()
        if current_obs_pose is None or current_goal_pose is None:
            return

        # TODO: get the control velocity command
        # --- State transitions ---
        if self.state == 'go_to_goal':
            dist_to_goal = math.hypot(current_goal_pose[0], current_goal_pose[1])
            if dist_to_goal < 0.3:
                self.get_logger().info('Goal reached! Returning to start.')
                self.state = 'return_to_start'

        elif self.state == 'return_to_start':
            if self.start_pose is not None:
                rx, ry, _ = self.get_robot_pose_in_odom()
                if rx is not None:
                    dist_to_start = math.hypot(rx - self.start_pose[0], ry - self.start_pose[1])
                    if dist_to_start < 0.3:
                        self.get_logger().info('Returned to start. Stopping.')
                        self.pub_control_cmd.publish(Twist())
                        return

        cmd_vel = self.controller(current_obs_pose, current_goal_pose)
        self.pub_control_cmd.publish(cmd_vel)
        #################################################

    # ------------------------------------------------------------------
    def controller(self, current_obs_pose, current_goal_pose):
        # Instructions: You can implement your own control algorithm here.

        ########### Write your code here ###########

        # TODO: Update the control velocity command
        #
        # Artificial Potential Field (APF) controller
        # -------------------------------------------------
        # Attractive potential  : pulls robot toward the active target.
        # Repulsive potential   : pushes robot away from the obstacle
        #                         when inside influence radius d0.
        # The net force vector is converted to (linear.x, angular.z).
        #
        # All 2-D positions are in the robot base_footprint frame
        # (x = forward, y = left).

        # Tuning knobs ---------------------------------------------------
        K_att       = 1.0   # Attractive gain
        K_rep       = 0.5   # Repulsive gain
        d0          = 0.8   # Obstacle influence radius [m]
        max_linear  = 0.3   # Max forward speed [m/s]
        max_angular = 1.0   # Max yaw rate [rad/s]
        K_angular   = 2.0   # Proportional gain for heading error
        # ----------------------------------------------------------------

        # Determine active target in base_footprint frame
        if self.state == 'go_to_goal':
            target = np.array([current_goal_pose[0], current_goal_pose[1]])
        else:
            # Transform odom-frame start pose into base_footprint frame
            rx, ry, ryaw = self.get_robot_pose_in_odom()
            if rx is None or self.start_pose is None:
                return Twist()
            dx = self.start_pose[0] - rx
            dy = self.start_pose[1] - ry
            target = np.array([
                 math.cos(ryaw) * dx + math.sin(ryaw) * dy,
                -math.sin(ryaw) * dx + math.cos(ryaw) * dy
            ])

        # Attractive force (unit vector toward target, scaled by K_att)
        dist_target = np.linalg.norm(target)
        F_att = K_att * (target / dist_target) if dist_target > 1e-4 else np.zeros(2)

        # Repulsive force (away from obstacle when inside d0)
        obs_xy   = np.array([current_obs_pose[0], current_obs_pose[1]])
        dist_obs = np.linalg.norm(obs_xy)
        if 1e-4 < dist_obs < d0:
            rep_mag = K_rep * (1.0 / dist_obs - 1.0 / d0) / (dist_obs ** 2)
            F_rep   = rep_mag * (-obs_xy / dist_obs)
        else:
            F_rep = np.zeros(2)

        # Combined force -> desired heading
        F_total       = F_att + F_rep
        desired_angle = math.atan2(F_total[1], F_total[0])

        # Angular velocity proportional to heading error
        angular_z = max(-max_angular, min(max_angular, K_angular * desired_angle))

        # Linear velocity: slow down for large heading error or near obstacle
        angle_factor = math.cos(desired_angle)
        obs_factor   = min(1.0, max(0.0, (dist_obs - 0.3) / (d0 - 0.3))) if dist_obs < d0 else 1.0
        linear_x     = max_linear * max(0.0, angle_factor) * obs_factor

        cmd_vel = Twist()
        cmd_vel.linear.x  = linear_x
        cmd_vel.linear.y  = 0.0
        cmd_vel.angular.z = angular_z
        return cmd_vel

        ############################################

def main(args=None):
    rclpy.init(args=args)
    tracking_node = TrackingNode()
    rclpy.spin(tracking_node)
    tracking_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
