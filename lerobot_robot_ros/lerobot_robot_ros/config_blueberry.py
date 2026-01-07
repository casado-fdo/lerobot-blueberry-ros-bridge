from dataclasses import dataclass, field

from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig
from lerobot.cameras.configs import ColorMode, Cv2Rotation
from lerobot.robots import RobotConfig

import os

@RobotConfig.register_subclass("blueberry")
@dataclass
class BlueberryROSConfig(RobotConfig):
    """Configuration for Blueberry robot."""

    # Cameras configuration
    left_camera_config = RealSenseCameraConfig(
        serial_number_or_name=os.getenv("LEFT_RS_SERIAL_NO"), 
        fps=int(os.getenv("RECORDING_FPS", "30")),
        warmup_s=0,
        width=640,
        height=480,
        color_mode=ColorMode.RGB,
        use_depth=False, # Depth is not supported yet by lerobot
        rotation=Cv2Rotation.ROTATE_180
    )
    right_camera_config = RealSenseCameraConfig(
        serial_number_or_name=os.getenv("RIGHT_RS_SERIAL_NO"), 
        fps=int(os.getenv("RECORDING_FPS", "30")),
        warmup_s=0,
        width=640,
        height=480,
        color_mode=ColorMode.RGB,
        use_depth=False, # Depth is not supported yet by lerobot
        rotation=Cv2Rotation.NO_ROTATION
    )
    cameras: dict[str, RealSenseCameraConfig] = field(default_factory=lambda: {"left": BlueberryROSConfig.left_camera_config, "right": BlueberryROSConfig.right_camera_config})
    
    
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
        "left_kinova_j7",
        "right_kinova_j1",
        "right_kinova_j2",
        "right_kinova_j3",
        "right_kinova_j4",
        "right_kinova_j5",
        "right_kinova_j6",
        "right_kinova_j7",
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
        1.0,
        ]
    )

    right_arm_teleop_topic: str = "/r_kinova_/lerobot/cartesian_velocity"
    left_arm_teleop_topic: str = "/l_kinova_/lerobot/cartesian_velocity"
    robot_joint_state_pos_topic: str = "/blueberry/joint_state/positions"
    robot_joint_state_vel_topic: str = "/blueberry/joint_state/velocities" 