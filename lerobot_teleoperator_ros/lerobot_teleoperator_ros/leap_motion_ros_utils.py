import numpy as np
import rospy
import os
from geometry_msgs.msg import TwistStamped

os.environ['ROS_PYTHON_LOG_CONFIG_FILE'] = '|'  # specify dummy file

class LeapMotionROSInterface():
    """Stream hand pose commands (left and right) from leap motion input."""

    def __init__(self, left_arm_topic: str, right_arm_topic: str):
        # Initialize ROS node
        if not rospy.get_node_uri():
            rospy.init_node('lerobot_teleop_ros_interface', anonymous=True)

        # Subscribe to the teleoperation data
        self.sub_arm1_teleop = rospy.Subscriber(left_arm_topic, TwistStamped, self.arm1_teleop_callback)
        self.sub_arm2_teleop = rospy.Subscriber(right_arm_topic, TwistStamped, self.arm2_teleop_callback)

        self.last_left_vel_command = TwistStamped()
        self.last_right_vel_command = TwistStamped()


    def get_latest_data(self) -> np.ndarray:
        # Return the latest data as a numpy array
        left_linear = [self.last_left_vel_command.twist.linear.x,
                       self.last_left_vel_command.twist.linear.y,
                       self.last_left_vel_command.twist.linear.z]
        left_angular = [self.last_left_vel_command.twist.angular.x,
                        self.last_left_vel_command.twist.angular.y,
                        self.last_left_vel_command.twist.angular.z]
        right_linear = [self.last_right_vel_command.twist.linear.x,
                        self.last_right_vel_command.twist.linear.y,
                        self.last_right_vel_command.twist.linear.z]
        right_angular = [self.last_right_vel_command.twist.angular.x,
                         self.last_right_vel_command.twist.angular.y,
                         self.last_right_vel_command.twist.angular.z]

        # Reset the last commands after reading
        self.last_left_vel_command = TwistStamped()
        self.last_right_vel_command = TwistStamped()

        return np.array(left_linear + left_angular + right_linear + right_angular)


    def arm1_teleop_callback(self, data):
        # Get the position and orientation of the hand
        self.last_left_vel_command = data
    

    def arm2_teleop_callback(self, data):
        # Get the position and orientation of the hand
        self.last_right_vel_command = data


    def stop(self):
        # Stop the interface
        rospy.signal_shutdown("Leap Motion ROS Interface stopped.")