import numpy as np
import rospy
import os
from geometry_msgs.msg import TwistStamped
from std_msgs.msg import Int32MultiArray

os.environ['ROS_PYTHON_LOG_CONFIG_FILE'] = '|'  # specify dummy file

class LeapMotionROSInterface():
    """Stream hand pose commands (left and right) from leap motion input."""

    def __init__(self, left_arm_topic: str, right_arm_topic: str, left_hand_topic: str = None, right_hand_topic: str = None):
        # Initialize ROS node
        if not rospy.get_node_uri():
            rospy.init_node('lerobot_teleop_ros_interface', anonymous=True)

        # Subscribe to the teleoperation data
        self.sub_l_arm_teleop = rospy.Subscriber(left_arm_topic, TwistStamped, self.l_arm_teleop_callback)
        self.sub_r_arm_teleop = rospy.Subscriber(right_arm_topic, TwistStamped, self.r_arm_teleop_callback)
        self.sub_l_hand_teleop = rospy.Subscriber(left_hand_topic, Int32MultiArray, self.l_hand_teleop_callback)
        self.sub_r_hand_teleop = rospy.Subscriber(right_hand_topic, Int32MultiArray, self.r_hand_teleop_callback)

        self.last_l_arm_vel_command = TwistStamped()
        self.last_r_arm_vel_command = TwistStamped()
        self.last_l_hand_pos_command = Int32MultiArray()
        self.last_l_hand_pos_command.data = [0.] * 6
        self.last_r_hand_pos_command = Int32MultiArray()
        self.last_r_hand_pos_command.data = [0.] * 6


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

        # Reset the last commands after reading
        self.last_l_arm_vel_command = TwistStamped()
        self.last_r_arm_vel_command = TwistStamped()

        return np.array(l_arm_linear + l_arm_angular + l_hand_positions + r_arm_linear + r_arm_angular + r_hand_positions)


    def l_arm_teleop_callback(self, data):
        # Get the position and orientation of the end effector
        self.last_l_arm_vel_command = data
    

    def r_arm_teleop_callback(self, data):
        # Get the position and orientation of the end effector
        self.last_r_arm_vel_command = data


    def l_hand_teleop_callback(self, data):
        # Get the position for each of the fingers
        self.last_l_hand_pos_command = data


    def r_hand_teleop_callback(self, data):
        # Get the position for each of the fingers
        self.last_r_hand_pos_command = data        
        

    def stop(self):
        # Stop the interface
        rospy.signal_shutdown("Leap Motion ROS Interface stopped.")