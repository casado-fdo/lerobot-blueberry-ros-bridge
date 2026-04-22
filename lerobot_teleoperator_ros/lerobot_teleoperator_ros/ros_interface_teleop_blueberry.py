import numpy as np
import rospy
import os
import time
from geometry_msgs.msg import TwistStamped
from std_msgs.msg import Int32MultiArray, Float32MultiArray

os.environ['ROS_PYTHON_LOG_CONFIG_FILE'] = '|'  # specify dummy file

class BlueberryTeleopROSInterface():
    """Stream hand pose commands (left and right) from leap motion input and joy commands for base control."""

    def __init__(self, 
        left_arm_topic: str, 
        right_arm_topic: str, 
        left_hand_topic: str = None, 
        right_hand_topic: str = None,
        base_topic: str = None,
        device_timeout_sec: float = 1.0,
    ):
        # Initialize ROS node
        if not rospy.get_node_uri():
            rospy.init_node('lerobot_teleop_ros_interface', anonymous=True)

        # Subscribe to the teleoperation data
        self.sub_l_arm_teleop = rospy.Subscriber(left_arm_topic, TwistStamped, self.l_arm_teleop_callback)
        self.sub_r_arm_teleop = rospy.Subscriber(right_arm_topic, TwistStamped, self.r_arm_teleop_callback)
        self.sub_l_hand_teleop = rospy.Subscriber(left_hand_topic, Int32MultiArray, self.l_hand_teleop_callback)
        self.sub_r_hand_teleop = rospy.Subscriber(right_hand_topic, Int32MultiArray, self.r_hand_teleop_callback)
        self.sub_base_teleop = rospy.Subscriber(base_topic, Float32MultiArray, self.base_teleop_callback)

        self.last_l_arm_vel_command = TwistStamped()
        self.last_r_arm_vel_command = TwistStamped()
        self.last_l_hand_pos_command = Int32MultiArray()
        self.last_l_hand_pos_command.data = [0.] * 6
        self.last_r_hand_pos_command = Int32MultiArray()
        self.last_r_hand_pos_command.data = [0.] * 6
        self.last_base_joy_command = Float32MultiArray()
        self.last_base_joy_command.data = [0.] * 2
        
        # Device health monitoring
        self.device_timeout_sec = device_timeout_sec
        self.last_l_arm_time = 0.0
        self.last_r_arm_time = 0.0
        self.last_l_hand_time = 0.0
        self.last_r_hand_time = 0.0
        self.last_base_time = 0.0

    def get_latest_data(self) -> np.ndarray:
        # Return the latest data as a numpy array
        l_arm_linear = [self.last_l_arm_vel_command.twist.linear.x,
                       self.last_l_arm_vel_command.twist.linear.y,
                       self.last_l_arm_vel_command.twist.linear.z]
        l_arm_angular = [self.last_l_arm_vel_command.twist.angular.x,
                        self.last_l_arm_vel_command.twist.angular.y,
                        self.last_l_arm_vel_command.twist.angular.z]
        l_hand_positions = list(self.last_l_hand_pos_command.data)            
        r_arm_linear = [self.last_r_arm_vel_command.twist.linear.x,
                        self.last_r_arm_vel_command.twist.linear.y,
                        self.last_r_arm_vel_command.twist.linear.z]
        r_arm_angular = [self.last_r_arm_vel_command.twist.angular.x,
                         self.last_r_arm_vel_command.twist.angular.y,
                         self.last_r_arm_vel_command.twist.angular.z]
        r_hand_positions = list(self.last_r_hand_pos_command.data)
        base_joy = [self.last_base_joy_command.data[0],
                       self.last_base_joy_command.data[1]]

        # Reset the last arm commands after reading
        self.last_l_arm_vel_command = TwistStamped()
        self.last_r_arm_vel_command = TwistStamped()

        return np.array(l_arm_linear + l_arm_angular + l_hand_positions + r_arm_linear + r_arm_angular + r_hand_positions + base_joy)


    def l_arm_teleop_callback(self, data):
        # Get the position and orientation of the end effector
        self.last_l_arm_vel_command = data
        self.last_l_arm_time = time.time()
    

    def r_arm_teleop_callback(self, data):
        # Get the position and orientation of the end effector
        self.last_r_arm_vel_command = data
        self.last_r_arm_time = time.time()


    def l_hand_teleop_callback(self, data):
        # Get the position for each of the fingers
        self.last_l_hand_pos_command = data
        self.last_l_hand_time = time.time()


    def r_hand_teleop_callback(self, data):
        # Get the position for each of the fingers
        self.last_r_hand_pos_command = data
        self.last_r_hand_time = time.time()        
        

    def base_teleop_callback(self, data):
        # Get the base joy command
        self.last_base_joy_command = data
        self.last_base_time = time.time()
        
    def is_device_alive(self) -> bool:
        """Check if the device (Leap Motion controller) is still alive and responsive.
        
        Returns:
            bool: True if any teleoperation topic has received data within the timeout period, False otherwise.
        """
        current_time = time.time()
        
        # Check if any of the teleoperation topics have received recent data
        # We consider the device alive if ANY of the following topics has recent activity
        if (current_time - self.last_l_arm_time < self.device_timeout_sec or
            current_time - self.last_r_arm_time < self.device_timeout_sec or
            current_time - self.last_l_hand_time < self.device_timeout_sec or
            current_time - self.last_r_hand_time < self.device_timeout_sec):
            return True
        
        return False
    
    def stop(self):
        # Stop the interface
        rospy.signal_shutdown("Blueberry Teleoperation ROS Interface stopped.")

    