import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, PoseStamped
from tf2_ros import TransformException, Buffer, TransformListener
import numpy as np
import math
import time
from std_msgs.msg import Bool

## Functions for quaternion and rotation matrix conversion
## The code is adapted from the general_robotics_toolbox package
## Code reference: https://github.com/rpiRobotics/rpi_general_robotics_toolbox_py
def hat(k):
    """
    Returns a 3 x 3 cross product matrix for a 3 x 1 vector

             [  0 -k3  k2]
     khat =  [ k3   0 -k1]
             [-k2  k1   0]

    :type    k: numpy.array
    :param   k: 3 x 1 vector
    :rtype:  numpy.array
    :return: the 3 x 3 cross product matrix
    """

    khat=np.zeros((3,3))
    khat[0,1]=-k[2]
    khat[0,2]=k[1]
    khat[1,0]=k[2]
    khat[1,2]=-k[0]
    khat[2,0]=-k[1]
    khat[2,1]=k[0]
    return khat

def q2R(q):
    """
    Converts a quaternion into a 3 x 3 rotation matrix according to the
    Euler-Rodrigues formula.
    
    :type    q: numpy.array
    :param   q: 4 x 1 vector representation of a quaternion q = [q0;qv]
    :rtype:  numpy.array
    :return: the 3x3 rotation matrix    
    """
    
    I = np.identity(3)
    qhat = hat(q[1:4])
    qhat2 = qhat.dot(qhat)
    return I + 2*q[0]*qhat + 2*qhat2
######################

def euler_from_quaternion(q):
    w=q[0]
    x=q[1]
    y=q[2]
    z=q[3]
    # euler from quaternion
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
        
        self.state = "Goal"
        self.patrol_num = 0
        self.patrol_points = [ np.array([1, 0, 0]),np.array([1, -1, 0]),np.array([0, -1, 0]),np.array([0, 0, 0])]
        self.patrol = False
        self.start = None

        #EMERSON ADD
        self.charge_point = None
        self.go_charge = False
        ##

        # ROS parameters
        self.declare_parameter('world_frame_id', 'odom')

        # Create a transform listener
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
        # Create publisher for the control command
        self.pub_control_cmd = self.create_publisher(Twist, '/track_cmd_vel', 10)
        # Create a subscriber to the detected object pose
        self.sub_detected_goal_pose = self.create_subscription(PoseStamped, 'detected_color_object_pose', self.detected_obs_pose_callback, 10)
        self.sub_detected_obs_pose = self.create_subscription(PoseStamped, 'detected_color_goal_pose', self.detected_goal_pose_callback, 10)

        #EMERSON ADD
        self.sub_go_charge = self.create_subscription(
            Bool,
            'go_charge',
            self.go_charge_callback,
            10
        )

        self.sub_patrol = self.create_subscription(
            Bool,
            'patrol',
            self.patrol_callback,
            10
        )
        ##
        # Create timer, running at 100Hz
        self.timer = self.create_timer(0.01, self.timer_update)

        self.state = "Goal"
        self.start = None
    
    def detected_obs_pose_callback(self, msg):
        #self.get_logger().info('Received Detected Object Pose')
        
        odom_id = self.get_parameter('world_frame_id').get_parameter_value().string_value
        center_points = np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z])
        
        # TODO: Filtering
        # You can decide to filter the detected object pose here
        # For example, you can filter the pose based on the distance from the camera
        # or the height of the object
        # if np.linalg.norm(center_points) > 3 or center_points[2] > 0.7:
        #     return
        ############################################################ edit 1
        if np.linalg.norm(center_points) > 3 or center_points[2] > 0.7:
            return

        print("Found Obstacle")
        ##########################################################
        
        try:
            # Transform the center point from the camera frame to the world frame
            transform = self.tf_buffer.lookup_transform(odom_id,msg.header.frame_id,rclpy.time.Time(),rclpy.duration.Duration(seconds=0.1))
            t_R = q2R(np.array([transform.transform.rotation.w,transform.transform.rotation.x,transform.transform.rotation.y,transform.transform.rotation.z]))
            cp_world = t_R@center_points+np.array([transform.transform.translation.x,transform.transform.translation.y,transform.transform.translation.z])
        except TransformException as e:
            self.get_logger().error('Transform Error: {}'.format(e))
            return
        
        # Get the detected object pose in the world frame
        self.obs_pose = cp_world

    def detected_goal_pose_callback(self, msg):
        #self.get_logger().info('Received Detected Object Pose')
        
        odom_id = self.get_parameter('world_frame_id').get_parameter_value().string_value
        center_points = np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z])
        
        # TODO: Filtering
        # You can decide to filter the detected object pose here
        # For example, you can filter the pose based on the distance from the camera
        # or the height of the object
        # if np.linalg.norm(center_points) > 3 or center_points[2] > 0.7:
        #     return

        ################################################################
        if np.linalg.norm(center_points) > 3 or center_points[2] > 0.7:
            return
        
        print("Found Goal")
        ###############################################################
        
        try:
            # Transform the center point from the camera frame to the world frame
            transform = self.tf_buffer.lookup_transform(odom_id,msg.header.frame_id,rclpy.time.Time(),rclpy.duration.Duration(seconds=0.1))
            t_R = q2R(np.array([transform.transform.rotation.w,transform.transform.rotation.x,transform.transform.rotation.y,transform.transform.rotation.z]))
            cp_world = t_R@center_points+np.array([transform.transform.translation.x,transform.transform.translation.y,transform.transform.translation.z])
        except TransformException as e:
            self.get_logger().error('Transform Error: {}'.format(e))
            return
        
        # Get the detected object pose in the world frame
        if self.state == "Goal":
            self.goal_pose = cp_world
        
    def get_current_poses(self):
        
        odom_id = self.get_parameter('world_frame_id').get_parameter_value().string_value
        # Get the current robot pose
        try:
            # from base_footprint to odom
            transform = self.tf_buffer.lookup_transform('base_footprint', odom_id, rclpy.time.Time())
            self.robot_world_x = transform.transform.translation.x
            self.robot_world_y = transform.transform.translation.y
            self.robot_world_z = transform.transform.translation.z
            self.robot_world_R = q2R([transform.transform.rotation.w, transform.transform.rotation.x, transform.transform.rotation.y, transform.transform.rotation.z])
            self.robot_world_R_euler = euler_from_quaternion([transform.transform.rotation.w, transform.transform.rotation.x, transform.transform.rotation.y, transform.transform.rotation.z])

            
            ################################################################### changing this

            #EMERSON MOVE
            if self.start is None:
                self.start = np.array([self.robot_world_x, self.robot_world_y, self.robot_world_z])
                #EMERSON ADD
                self.charge_point = np.array([0.0, -1.0, 0.0])
            ##

            obstacle_pose = self.obs_pose
            goal_pose = self.goal_pose
            
            
            ################################################################### changing this ^
        
        except TransformException as e:
            self.get_logger().error('Transform error: ' + str(e))
            return
        
        return obstacle_pose, goal_pose

    
    #EMERSON ADD
    def go_charge_callback(self, msg):
        self.go_charge = msg.data
        print("go_charge:", self.go_charge)
        self.state = "Charge"
        if not self.go_charge:
            self.state = "Goal"
        
    def patrol_callback(self, msg):
        self.patrol = msg.data
        print("patrol:", self.patrol)
        if self.go_charge:
            self.state = "patrol"
            self.patrol_num = 0
            self.goal_pose = self.patrol_points[0]
        else:
            self.state = None
            self.state = "Goal"
    ##

    
    def timer_update(self):
        ################### Write your code here ###################
        
        ################################################################
        pose_check = self.get_current_poses()
        if pose_check is None:
            cmd_vel = Twist()
            cmd_vel.linear.x = 0.0
            cmd_vel.angular.z = 0.0
            self.pub_control_cmd.publish(cmd_vel)
            return
            
        current_obs_pose, current_goal_pose = pose_check

        #EMERSON ADD
        if self.go_charge and self.charge_point is not None:
            current_goal_pose = self.charge_point
        ###############################################################
        
        # Now, the robot stops if the object is not detected
        # But, you may want to think about what to do in this case
        # and update the command velocity accordingly
        if current_goal_pose is None:
            cmd_vel = Twist()
            cmd_vel.linear.x = 0.0
            cmd_vel.angular.z = 0.0
            self.pub_control_cmd.publish(cmd_vel)
            return
        
        # Get the current object pose in the robot base_footprint frame
        #current_obs_pose, current_goal_pose = self.get_current_poses()
        # ^commenting this line out
        
        # TODO: get the control velocity command
        ###################################################################### _        
        #cmd_vel = self.controller()   #old line
        odom_id = self.get_parameter('world_frame_id').get_parameter_value().string_value
        # Get the current robot pose
        # from base_footprint to odom
        transform = self.tf_buffer.lookup_transform('base_footprint', odom_id, rclpy.time.Time())
        self.robot_world_x = transform.transform.translation.x
        self.robot_world_y = transform.transform.translation.y
        self.robot_world_z = transform.transform.translation.z
        self.robot_world_R = q2R([transform.transform.rotation.w, transform.transform.rotation.x, transform.transform.rotation.y, transform.transform.rotation.z])


        ### EMERSON ADD LARGE CHUNK
        if self.go_charge and self.charge_point is not None:
            current_goal_pose = self.charge_point
        
            dist_to_charge = np.linalg.norm(
                self.charge_point[:2] - np.array([self.robot_world_x, self.robot_world_y])
            )
        
            if dist_to_charge < 0.10:
                cmd_vel = Twist()
                cmd_vel.linear.x = 0.0
                cmd_vel.linear.y = 0.0
                cmd_vel.angular.z = 0.0
                self.pub_control_cmd.publish(cmd_vel)
                self.go_charge = False
                print("Reached charging point")
                return
        ##
        
        cmd_vel = self.controller(current_obs_pose, current_goal_pose)

        """  EMERSON MOVE TO EARLIER PART
        if self.start is None:
            self.start = np.array([self.robot_world_x, self.robot_world_y, self.robot_world_z])
            #EMERSON ADD
            self.charge_point = self.start + np.array([0.0, -1, 0.0])
            ##
        """ 
        ################################################################### ^
        
        # publish the control command
        self.pub_control_cmd.publish(cmd_vel)
        #################################################
    
    def controller(self, obs_pose, goal_pose):
        # Instructions: You can implement your own control algorithm here
        # feel free to modify the code structure, add more parameters, more input variables for the function, etc.
        
        ########### Write your code here ###########

        ###################################################################### _
        # old code:
        # TODO: Update the control velocity command
        #cmd_vel = Twist()
        #cmd_vel.linear.x = 0
        #cmd_vel.linear.y = 0
        #cmd_vel.angular.z = 0
        #return cmd_vel

        #new code:
        K_v = 0.5
        K_h = 0.3
        zetta = 0.5
        n = 2
        Q = 1

        # EMERSON CHANGE (BACK TO ORIGINAL)
        pose = np.array([-self.robot_world_x, -self.robot_world_y, self.robot_world_z])
        #pose = np.array([self.robot_world_x, self.robot_world_y, self.robot_world_z])
        print("pose:", pose)
        world_goal_pose = None
        if self.state == "patrol":
            world_goal_pose = self.patrol_points[self.patrol_num]

        elif self.state == "Charge":
            world_goal_pose = self.charge_point

        else:
            #world_goal_pose = self.robot_world_R@self.goal_pose+np.array([self.robot_world_x,self.robot_world_y,self.robot_world_z])
            world_goal_pose = goal_pose
        print("goal:", world_goal_pose)

        dis_goal = (world_goal_pose - pose) 
        if self.state == "patrol":
            if dis_goal < 0.2:
                self.patrol_num = (self.patrol_num + 1) % 4

        #Potential Field
        U_grad = zetta * dis_goal
        #print(dis_goal)

        # EMERSON want to maybe change if line to "if not obs_pose is None:"

        """
        if not goal_pose is None:
            #world_obs_pose = self.robot_world_R@self.goal_pose+np.array([self.robot_world_x,self.robot_world_y,self.robot_world_z])
            #EMERSON want to maybe change if line to "world_obs_pose = obs_pose"
            world_obs_pose = goal_pose
            print("obs:", world_obs_pose)
            dis_obj = pose - world_obs_pose
            radius = 0.1
            if np.linalg.norm(dis_obj) - radius < Q:
                U_grad = U_grad - 0.5*n*(1/Q - 1/(np.linalg.norm(dis_obj)-radius))*1/(np.linalg.norm(dis_obj)-radius)**2*dis_obj/(np.linalg.norm(dis_obj))
        """

        if not obs_pose is None:
            world_obs_pose = obs_pose
            print("obs:", world_obs_pose)
            dis_obj = pose - world_obs_pose
            radius = 0.1
            if np.linalg.norm(dis_obj) - radius < Q:
                U_grad = U_grad - 0.5*n*(1/Q - 1/(np.linalg.norm(dis_obj)-radius))*1/(np.linalg.norm(dis_obj)-radius)**2*dis_obj/(np.linalg.norm(dis_obj))
        
        print(U_grad)
        #U_grad = self.robot_world_R@U_grad
        theta_star = np.arctan2(dis_goal[1],dis_goal[0])
        
        print(theta_star - self.robot_world_R_euler[2])
        gamma_star = max(-np.pi/2, min(np.pi/2, K_h * (theta_star - self.robot_world_R_euler[2])))
        v_star = np.array([0,0])

        if abs(gamma_star) < 0.1:
            v_star = np.array([min(2, max(-2, K_v *U_grad[0])),min(2, max(-2, K_v *U_grad[1]))])
        
        #gamma_star = 0.0
        
        delta_t = 0.01

        u = [float(v_star[0]),
            float(v_star[1]),
            float(gamma_star)]
        cmd_vel = Twist()
        cmd_vel.linear.x = u[0]
        cmd_vel.linear.y = u[1]
        cmd_vel.angular.z = u[2]
        
        self.get_logger().warn(f'{cmd_vel}')
        return cmd_vel
    
       ################################################################### ^


def main(args=None):
    # Initialize the rclpy library
    rclpy.init(args=args) 
    # Create the node
    tracking_node = TrackingNode()
    rclpy.spin(tracking_node)
    # Destroy the node explicitly
    tracking_node.destroy_node()
    # Shutdown the ROS client library for Python
    rclpy.shutdown()

if __name__ == '__main__':
    main()
