from dataclasses import dataclass, field

from lerobot.cameras import CameraConfig
from lerobot.robots import RobotConfig


@dataclass
class BlueberryInterfaceConfig:
    # Namespace used by ros_control nodes
    namespace: str = ""

    arm_joint_names: list[str] = field(
        default_factory=lambda: [
            "joint_1",
            "joint_2",
            "joint_3",
            "joint_4",
            "joint_5",
            "joint_6",
        ]
    )

    # Base link
    base_link: str = "base_link"

    # Only applicable if position control is used.
    min_joint_positions: list[float] | None = None
    max_joint_positions: list[float] | None = None


@dataclass
class BlueberryConfig(RobotConfig):
    # cameras
    cameras: dict[str, CameraConfig] = field(default_factory=dict)

    # ROS interface configuration
    ros_interface: BlueberryInterfaceConfig = field(default_factory=BlueberryInterfaceConfig)


@RobotConfig.register_subclass("blueberry")
@dataclass
class BlueberryRobotConfig(BlueberryConfig):
    """Configuration for Blueberry robot."""
    ros_interface: BlueberryInterfaceConfig = field(default_factory=BlueberryInterfaceConfig)