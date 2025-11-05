import rospy
import time 

from lerobot.utils.errors import DeviceNotConnectedError
from geometry_msgs.msg import TwistStamped
from std_msgs.msg import Float32MultiArray

from .config import ROSInterfaceConfig


class ROSInterface:
    """Class to interface with our custom Blueberry robot (ROS Noetic)."""

    def __init__(self, config: ROSInterfaceConfig):
        self.config = config
        self.robot_node = None
        self.r_kinova_teleop_cmd_pub = None

        self.is_connected = False
        self._last_joint_state = None

    def connect(self) -> None:
        if not rospy.get_node_uri():
            rospy.init_node("lerobot_ros_interface_node", anonymous=True)

        self.r_kinova_teleop_cmd_pub = rospy.Publisher("/r_kinova_/lerobot/cartesian_velocity", TwistStamped, queue_size=10) # TODO: pass them as parameters
        self.l_kinova_teleop_cmd_pub = rospy.Publisher("/l_kinova_/lerobot/cartesian_velocity", TwistStamped, queue_size=10)
        self.joint_state_pos_sub = rospy.Subscriber("/blueberry/joint_state/positions", Float32MultiArray, self._joint_state_pos_callback)
        self.joint_state_vel_sub = rospy.Subscriber("/blueberry/joint_state/velocities", Float32MultiArray, self._joint_state_vel_callback)

        time.sleep(2) # Give some time to connect to services and receive messages

        self.is_connected = True


    #def send_joint_position_command(self, joint_positions: list[float], unnormalize: bool = True) -> None:
    #    if not self.is_connected:
    #        raise DeviceNotConnectedError("ROSInterface is not connected. You need to call `connect()`.")
    #
    #    if unnormalize:
    #        if self.config.min_joint_positions is None or self.config.max_joint_positions is None:
    #            raise ValueError("Joint position normalization requires min and max joint positions to be set.")
    #        joint_positions = [
    #            min(max(pos, min_pos), max_pos)
    #            for pos, min_pos, max_pos in zip(joint_positions, self.config.min_joint_positions, self.config.max_joint_positions)
    #        ]
    #
    #    if len(joint_positions) != len(self.config.arm_joint_names):
    #        raise ValueError(f"Expected {len(self.config.arm_joint_names)} joint positions, but got {len(joint_positions)}.")
    #
    #    if self.teleop_cmd_pub is None:
    #        raise DeviceNotConnectedError("Position command publisher is not initialized.")
    #    msg = Float64MultiArray()
    #    msg.data = joint_positions
    #    self.teleop_cmd_pub.publish(msg)


    def send_leap_command(self, leap_command: dict[str, float]) -> None:
        if not self.is_connected:
            raise DeviceNotConnectedError("ROSInterface is not connected. You need to call `connect()`.")
      
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
        for idx, joint_name in enumerate(self.config.arm_joint_names):
            positions[joint_name] = msg.data[idx]
        self._last_joint_state["position"] = positions


    def _joint_state_vel_callback(self, msg: Float32MultiArray) -> None:
        self._last_joint_state = self._last_joint_state or {}
        velocities = {}
        for idx, joint_name in enumerate(self.config.arm_joint_names):
            velocities[joint_name] = msg.data[idx]
        self._last_joint_state["velocity"] = velocities


    def disconnect(self):
        if self.joint_state_pos_sub:
            self.joint_state_pos_sub.unregister()
            self.joint_state_pos_sub = None
        if self.joint_state_vel_sub:
            self.joint_state_vel_sub.unregister()
            self.joint_state_vel_sub = None
        if self.r_kinova_teleop_cmd_pub:
            self.r_kinova_teleop_cmd_pub.unregister()
            self.r_kinova_teleop_cmd_pub = None

        self.is_connected = False
