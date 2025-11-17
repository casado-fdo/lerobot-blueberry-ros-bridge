from dataclasses import dataclass, field

from lerobot.cameras import CameraConfig
from lerobot.robots import RobotConfig


@RobotConfig.register_subclass("blueberry")
@dataclass
class BlueberryROSConfig(RobotConfig):
    """Configuration for Blueberry robot."""

    # Cameras configuration
    # cameras: dict[str, CameraConfig] = field(default_factory=dict)
    # camera_config = {"front": OpenCVCameraConfig(index_or_path=0, width=640, height=480, fps=FPS)}

    # ROS interface configuration
    namespace: str = "blueberry"
    
    arm_joint_names: list[str] = field(
       default_factory=lambda: [
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
        ]
    )
    
    base_link: str = "blueberry_base_link"

    min_joint_positions: list[float] = field(
        default_factory=lambda: [
        -1.0,
        -1.0,
        -1.0,
        -1.0,
        -1.0,
        -1.0,
        ]
    )
    max_joint_positions: list[float] = field(
        default_factory=lambda: [
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        ]
    )

    right_arm_teleop_topic: str = "/r_kinova_/lerobot/cartesian_velocity"
    left_arm_teleop_topic: str = "/l_kinova_/lerobot/cartesian_velocity"
    robot_joint_state_pos_topic: str = "/blueberry/joint_state/positions"
    robot_joint_state_vel_topic: str = "/blueberry/joint_state/velocities" 