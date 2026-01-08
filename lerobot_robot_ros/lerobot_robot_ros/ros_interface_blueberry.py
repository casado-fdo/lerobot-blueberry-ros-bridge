import rospy
import time 
import os

from lerobot.utils.errors import DeviceNotConnectedError
from geometry_msgs.msg import TwistStamped
from std_msgs.msg import Float32MultiArray

from .config_blueberry import BlueberryROSConfig

os.environ['ROS_PYTHON_LOG_CONFIG_FILE'] = '|'  # specify dummy file

class BlueberryROSInterface:
    """Class to interface with our custom Blueberry robot (ROS Noetic)."""

    def __init__(self, config: BlueberryROSConfig):
        self.config = config
        self.robot_node = None
        self.r_kinova_teleop_cmd_pub = None

        self.is_connected = False
        self._last_joint_state = None

    def connect(self) -> None:
        if not rospy.get_node_uri():
            rospy.init_node("lerobot_ros_interface_node", anonymous=True)

        self.r_kinova_teleop_cmd_pub = rospy.Publisher(self.config.right_arm_teleop_topic, TwistStamped, queue_size=10) 
        self.l_kinova_teleop_cmd_pub = rospy.Publisher(self.config.left_arm_teleop_topic, TwistStamped, queue_size=10)
        self.joint_state_pos_sub = rospy.Subscriber(self.config.robot_joint_state_pos_topic, Float32MultiArray, self._joint_state_pos_callback)
        self.joint_state_effort_sub = rospy.Subscriber(self.config.robot_joint_state_effort_topic, Float32MultiArray, self._joint_state_effort_callback)

        time.sleep(2) # Give some time to connect to services and receive messages

        self.is_connected = True


    def send_leap_command(self, leap_command: dict[str, float]) -> None:
        if not self.is_connected:
            raise DeviceNotConnectedError("BlueberryROSInterface is not connected. You need to call `connect()`.")
      
        if self.r_kinova_teleop_cmd_pub is None or self.l_kinova_teleop_cmd_pub is None:
            raise DeviceNotConnectedError("Kinova command publishers are not initialised.")
        r_msg = self.cmd_to_ros_twist(leap_command, prefix="right_")
        l_msg = self.cmd_to_ros_twist(leap_command, prefix="left_")

        self.r_kinova_teleop_cmd_pub.publish(r_msg)
        self.l_kinova_teleop_cmd_pub.publish(l_msg)


    def cmd_to_ros_twist(self, leap_command: dict[str, float], prefix: str = "") -> TwistStamped:
        msg = TwistStamped()
        msg.header.stamp = rospy.Time.now()
        msg.twist.linear.x = leap_command.get(f"{prefix}linear.x", 0.0)
        msg.twist.linear.y = leap_command.get(f"{prefix}linear.y", 0.0)
        msg.twist.linear.z = leap_command.get(f"{prefix}linear.z", 0.0)
        msg.twist.angular.x = leap_command.get(f"{prefix}angular.x", 0.0)
        msg.twist.angular.y = leap_command.get(f"{prefix}angular.y", 0.0)
        msg.twist.angular.z = leap_command.get(f"{prefix}angular.z", 0.0)
        return msg


    @property
    def joint_state(self) -> dict | None:
        return self._last_joint_state


    def _joint_state_pos_callback(self, msg: Float32MultiArray) -> None:
        self._last_joint_state = self._last_joint_state or {}
        positions = {}
        for idx, joint_name in enumerate(self.config.blueberry_joint_names):
            positions[joint_name] = msg.data[idx]
        self._last_joint_state["position"] = positions

    def _joint_state_effort_callback(self, msg: Float32MultiArray) -> None:
        self._last_joint_state = self._last_joint_state or {}
        effort = {}
        for idx, joint_name in enumerate(self.config.blueberry_joint_names):
            effort[joint_name] = msg.data[idx]
        self._last_joint_state["effort"] = effort
        

    def disconnect(self):
        if self.joint_state_pos_sub:
            self.joint_state_pos_sub.unregister()
            self.joint_state_pos_sub = None
        if self.joint_state_effort_sub:
            self.joint_state_effort_sub.unregister()
            self.joint_state_effort_sub = None
        if self.r_kinova_teleop_cmd_pub:
            self.r_kinova_teleop_cmd_pub.unregister()
            self.r_kinova_teleop_cmd_pub = None

        self.is_connected = False
