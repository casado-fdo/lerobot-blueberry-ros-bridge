import rospy
from lerobot.utils.errors import DeviceNotConnectedError
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray

from .config import BlueberryInterfaceConfig

class BlueberryInterface:
    """Class to interface with our custom Blueberry robot (ROS Noetic)."""

    def __init__(self, config: BlueberryInterfaceConfig):
        self.config = config
        self.robot_node = None
        self.pos_cmd_pub = None
        self.traj_cmd_pub = None
        self.is_connected = False
        self._last_joint_state = None

    def connect(self) -> None:
        rospy.init_node("ros_lerobot_blueberry_interface", anonymous=True)

        self.pos_cmd_pub = rospy.Publisher("/TODO", Float64MultiArray, queue_size=10)
        self.joint_state_sub = rospy.Subscriber("joint_states", JointState, self._joint_state_callback)
        self.is_connected = True

    def send_joint_position_command(self, joint_positions: list[float], unnormalize: bool = True) -> None:
        if not self.is_connected:
            raise DeviceNotConnectedError("BlueberryInterface is not connected. You need to call `connect()`.")

        if unnormalize:
            if self.config.min_joint_positions is None or self.config.max_joint_positions is None:
                raise ValueError("Joint position normalization requires min and max joint positions to be set.")
            joint_positions = [
                min(max(pos, min_pos), max_pos)
                for pos, min_pos, max_pos in zip(joint_positions, self.config.min_joint_positions, self.config.max_joint_positions)
            ]

        if len(joint_positions) != len(self.config.arm_joint_names):
            raise ValueError(f"Expected {len(self.config.arm_joint_names)} joint positions, but got {len(joint_positions)}.")

        if self.pos_cmd_pub is None:
            raise DeviceNotConnectedError("Position command publisher is not initialized.")
        msg = Float64MultiArray()
        msg.data = joint_positions
        self.pos_cmd_pub.publish(msg)

    @property
    def joint_state(self) -> dict | None:
        return self._last_joint_state

    def _joint_state_callback(self, msg: JointState) -> None:
        self._last_joint_state = self._last_joint_state or {}
        positions = {}
        velocities = {}
        name_to_index = {name: i for i, name in enumerate(msg.name)}
        for joint_name in self.config.arm_joint_names:
            idx = name_to_index.get(joint_name)
            if idx is None:
                raise ValueError(f"Joint '{joint_name}' not found in joint state.")
            positions[joint_name] = msg.position[idx]
            velocities[joint_name] = msg.velocity[idx]

        self._last_joint_state["position"] = positions
        self._last_joint_state["velocity"] = velocities

    def disconnect(self):
        if self.joint_state_sub:
            self.joint_state_sub.unregister()
            self.joint_state_sub = None
        if self.pos_cmd_pub:
            self.pos_cmd_pub.unregister()
            self.pos_cmd_pub = None

        self.is_connected = False
