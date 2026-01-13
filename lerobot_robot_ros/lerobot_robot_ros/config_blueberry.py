from dataclasses import dataclass, field

from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig
from lerobot.cameras.configs import ColorMode, Cv2Rotation
from lerobot.robots import RobotConfig
from math import pi
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
        width=320,
        height=240,
        color_mode=ColorMode.RGB,
        use_depth=False, # Depth is not supported yet by lerobot (TODO: add when available)
        rotation=Cv2Rotation.ROTATE_180
    )
    right_camera_config = RealSenseCameraConfig(
        serial_number_or_name=os.getenv("RIGHT_RS_SERIAL_NO"), 
        fps=int(os.getenv("RECORDING_FPS", "30")),
        warmup_s=0,
        width=320,
        height=240,
        color_mode=ColorMode.RGB,
        use_depth=False, # Depth is not supported yet by lerobot (TODO: add when available)
        rotation=Cv2Rotation.NO_ROTATION
    )
    cameras: dict[str, RealSenseCameraConfig] = field(
        default_factory=lambda: {
            "left": BlueberryROSConfig.left_camera_config, 
            "right": BlueberryROSConfig.right_camera_config
        })
    
    
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

    right_arm_teleop_topic: str = "/r_kinova_/lerobot/cartesian_velocity"
    left_arm_teleop_topic: str = "/l_kinova_/lerobot/cartesian_velocity"
    right_hand_teleop_topic: str = "/right_hand/lerobot/hand_angles"
    left_hand_teleop_topic: str = "/left_hand/lerobot/hand_angles"
    robot_joint_state_pos_topic: str = "/blueberry/joint_state/position"
    robot_joint_state_effort_topic: str = "/blueberry/joint_state/effort"


    # Normalisation parameters
    gen3_inf_joint_pos_lim = 2.97  # Approx 170 degrees, used to normalise joints 1,3,5,7 implemented as continuous/rotation hardware (-inf to +inf)
    gen3_big_joint_effort_lim = 39.0  # Nm
    gen3_small_joint_effort_lim = 9.0  # Nm
    gen3_min_joint_positions: list[float] = [-gen3_inf_joint_pos_lim, -2.25, -gen3_inf_joint_pos_lim, -2.58, -gen3_inf_joint_pos_lim, 2.10, -gen3_inf_joint_pos_lim]
    gen3_max_joint_positions: list[float] = [gen3_inf_joint_pos_lim, 2.25, gen3_inf_joint_pos_lim, 2.58, gen3_inf_joint_pos_lim, 2.10, gen3_inf_joint_pos_lim]
    gen3_min_joint_efforts: list[float] = [-gen3_big_joint_effort_lim, -gen3_big_joint_effort_lim, -gen3_big_joint_effort_lim, -gen3_big_joint_effort_lim, -gen3_small_joint_effort_lim, -gen3_small_joint_effort_lim, -gen3_small_joint_effort_lim]
    gen3_max_joint_efforts: list[float] = [gen3_big_joint_effort_lim, gen3_big_joint_effort_lim, gen3_big_joint_effort_lim, gen3_big_joint_effort_lim, gen3_small_joint_effort_lim, gen3_small_joint_effort_lim, gen3_small_joint_effort_lim]
    inspire_hand_min_joint_position: float = [0.0] * 6
    inspire_hand_max_joint_position: float = [1000.0] * 6
    inspire_hand_min_joint_effort: float = [-500.0] * 6
    inspire_hand_max_joint_effort: float = [2000.0] * 6
