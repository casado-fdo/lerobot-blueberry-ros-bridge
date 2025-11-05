from dataclasses import dataclass, field

from lerobot.cameras import CameraConfig
from lerobot.robots import RobotConfig


@dataclass
class ROSInterfaceConfig:
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
class ROSConfig(RobotConfig):
    # cameras
    cameras: dict[str, CameraConfig] = field(default_factory=dict)

    # ROS interface configuration
    ros_interface: ROSInterfaceConfig = field(default_factory=ROSInterfaceConfig)


@RobotConfig.register_subclass("blueberry")
@dataclass
class BlueberryROSConfig(ROSConfig):
    """Configuration for Blueberry robot."""
    ros_interface: ROSInterfaceConfig = field(
        default_factory=lambda: ROSInterfaceConfig(
            namespace="blueberry",
            arm_joint_names=[
                "left_kinova_j1",
                "left_kinova_j2",
                "left_kinova_j3",
                "left_kinova_j4",
                "left_kinova_j5",
                "left_kinova_j6",
                "right_kinova_j1",
                "right_kinova_j2",
                "right_kinova_j3",
                "right_kinova_j4",
                "right_kinova_j5",
                "right_kinova_j6",
            ],
            base_link="blueberry_base_link",
            min_joint_positions=[
                -1.0,
                -1.0,
                -1.0,
                -1.0,
                -1.0,
                -1.0,
            ],
            max_joint_positions=[
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
            ],
        ),
    )
