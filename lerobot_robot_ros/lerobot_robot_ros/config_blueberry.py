from dataclasses import dataclass, field

from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig
from lerobot.cameras.configs import ColorMode, Cv2Rotation
from lerobot.robots import RobotConfig
from math import pi
import os

KINOVA_MIN_JOINT_POSITION = -2.0 * pi
KINOVA_MAX_JOINT_POSITION = 2.0 * pi
INSPIRE_HAND_MIN_JOINT_POSITION = 0.0
INSPIRE_HAND_MAX_JOINT_POSITION = 1000.0

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
        use_depth=False, # Depth is not supported yet by lerobot (TODO: add when available)
        rotation=Cv2Rotation.ROTATE_180
    )
    right_camera_config = RealSenseCameraConfig(
        serial_number_or_name=os.getenv("RIGHT_RS_SERIAL_NO"), 
        fps=int(os.getenv("RECORDING_FPS", "30")),
        warmup_s=0,
        width=640,
        height=480,
        color_mode=ColorMode.RGB,
        use_depth=False, # Depth is not supported yet by lerobot (TODO: add when available)
        rotation=Cv2Rotation.NO_ROTATION
    )
    cameras: dict[str, RealSenseCameraConfig] = field(default_factory=lambda: {"left": BlueberryROSConfig.left_camera_config, "right": BlueberryROSConfig.right_camera_config})
    
    
    # ROS interface configuration
    namespace: str = "blueberry"
    
    blueberry_joint_names: list[str] = field(
       default_factory=lambda: [
        # Left arm joints
        "l_arm_j1", "l_arm_j2", "l_arm_j3", "l_arm_j4", "l_arm_j5", "l_arm_j6", "l_arm_j7",
        # Left hand joints
        "l_hand_pinky", "l_hand_ring", "l_hand_middle", "l_hand_index", "l_hand_thumb1", "l_hand_thumb2", 
        # Right arm joints
        "r_arm_j1", "r_arm_j2", "r_arm_j3", "r_arm_j4", "r_arm_j5", "r_arm_j6", "r_arm_j7",
        # Right hand joints
        "r_hand_pinky", "r_hand_ring", "r_hand_middle", "r_hand_index", "r_hand_thumb1", "r_hand_thumb2",
        ]
    )
    
    base_link: str = "blueberry_base_link"

    min_joint_positions: list[float] = field(
        default_factory=lambda: [
            # Left kinova arm joints
            [KINOVA_MIN_JOINT_POSITION] * 7,
            # Left inspire hand joints
            [INSPIRE_HAND_MIN_JOINT_POSITION] * 6,
            # Right kinova arm joints
            [KINOVA_MIN_JOINT_POSITION] * 7,
            # Right inspire hand joints
            [INSPIRE_HAND_MIN_JOINT_POSITION] * 6,
        ]
    )
    max_joint_positions: list[float] = field(
        default_factory=lambda: [
            # Left kinova arm joints
            [KINOVA_MAX_JOINT_POSITION] * 7,
            # Left inspire hand joints
            [INSPIRE_HAND_MAX_JOINT_POSITION] * 6,
            # Right kinova arm joints
            [KINOVA_MAX_JOINT_POSITION] * 7,
            # Right inspire hand joints
            [INSPIRE_HAND_MAX_JOINT_POSITION] * 6,
        ]
    )

    right_arm_teleop_topic: str = "/r_kinova_/lerobot/cartesian_velocity"
    left_arm_teleop_topic: str = "/l_kinova_/lerobot/cartesian_velocity"
    robot_joint_state_pos_topic: str = "/blueberry/joint_state/position"
    #robot_joint_state_vel_topic: str = "/blueberry/joint_state/velocities" 
    robot_joint_state_effort_topic: str = "/blueberry/joint_state/effort"