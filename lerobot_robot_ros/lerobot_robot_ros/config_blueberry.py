from dataclasses import dataclass, field

from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig
from lerobot.cameras.opencv.camera_opencv import OpenCVCamera
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.cameras.configs import ColorMode, Cv2Rotation, Cv2Backends,CameraConfig
from lerobot.robots import RobotConfig
import os

# Normalisation constants
GEN3_INF_JOINT_POS_LIM = 2.97  # Approx 170 degrees
GEN3_BIG_JOINT_EFFORT_LIM = 39.0  # Nm
GEN3_SMALL_JOINT_EFFORT_LIM = 9.0  # Nm


@RobotConfig.register_subclass("blueberry")
@dataclass
class BlueberryROSConfig(RobotConfig):
    """Configuration for Blueberry robot."""

    cam_width = int(os.getenv("RECORDING_VIDEO_WIDTH", "320")) # 1280, 960, 640, 320
    cam_height = int(os.getenv("RECORDING_VIDEO_HEIGHT", "240")) #  720, 540, 480, 240
    cam_fps = int(os.getenv("RECORDING_FPS", "15"))
    rs_fps = 30 if cam_width == 320 else cam_fps # The realsense can only run at 5, 30 or 60 fps in 320x240 resolution

    # Cameras configuration
    left_camera_config = RealSenseCameraConfig(
        serial_number_or_name=os.getenv("LEFT_RS_SERIAL_NO"), 
        fps=rs_fps,
        warmup_s=0,
        width=cam_width,
        height=cam_height,
        color_mode=ColorMode.RGB,
        use_depth=False, # Depth is not supported yet by lerobot (TODO: add when available)
        rotation=Cv2Rotation.ROTATE_180
    )
    right_camera_config = RealSenseCameraConfig(
        serial_number_or_name=os.getenv("RIGHT_RS_SERIAL_NO"), 
        fps=rs_fps,
        warmup_s=0,
        width=cam_width,
        height=cam_height,
        color_mode=ColorMode.RGB,
        use_depth=False, # Depth is not supported yet by lerobot (TODO: add when available)
        rotation=Cv2Rotation.NO_ROTATION
    )
    user_camera_config = OpenCVCameraConfig(
        index_or_path='/dev/video20',
        fps=cam_fps,
        width=cam_width,  
        height=cam_height,
        color_mode=ColorMode.RGB,
        backend=Cv2Backends.V4L2,
        warmup_s=1,
    )
    cameras: dict[str, CameraConfig] = field(
        default_factory=lambda: {
            "left": BlueberryROSConfig.left_camera_config, 
            "right": BlueberryROSConfig.right_camera_config,
            "user": BlueberryROSConfig.user_camera_config,
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
    base_teleop_topic: str = "/rnet/lerobot/joy"
    robot_joint_state_pos_topic: str = "/blueberry/joint_state/position"
    robot_joint_state_effort_topic: str = "/blueberry/joint_state/effort"


    # Normalisation parameters
    gen3_min_joint_positions: list[float] = field(
        default_factory=lambda: [
        -GEN3_INF_JOINT_POS_LIM, -2.25, -GEN3_INF_JOINT_POS_LIM, -2.58, -GEN3_INF_JOINT_POS_LIM, 2.10, -GEN3_INF_JOINT_POS_LIM
        ]
    )
    gen3_max_joint_positions: list[float] = field(
        default_factory=lambda: [
            GEN3_INF_JOINT_POS_LIM, 2.25, GEN3_INF_JOINT_POS_LIM, 2.58, GEN3_INF_JOINT_POS_LIM, 2.10, GEN3_INF_JOINT_POS_LIM
        ]
    )
    gen3_min_joint_efforts: list[float] = field(
        default_factory=lambda: [
            -GEN3_BIG_JOINT_EFFORT_LIM, -GEN3_BIG_JOINT_EFFORT_LIM, -GEN3_BIG_JOINT_EFFORT_LIM, -GEN3_BIG_JOINT_EFFORT_LIM, -GEN3_SMALL_JOINT_EFFORT_LIM, -GEN3_SMALL_JOINT_EFFORT_LIM, -GEN3_SMALL_JOINT_EFFORT_LIM
        ]
    )
    gen3_max_joint_efforts: list[float] = field(
        default_factory=lambda: [
            GEN3_BIG_JOINT_EFFORT_LIM, GEN3_BIG_JOINT_EFFORT_LIM, GEN3_BIG_JOINT_EFFORT_LIM, GEN3_BIG_JOINT_EFFORT_LIM, GEN3_SMALL_JOINT_EFFORT_LIM, GEN3_SMALL_JOINT_EFFORT_LIM, GEN3_SMALL_JOINT_EFFORT_LIM
        ]
    )
    inspire_hand_min_joint_position: float = field(default_factory=lambda: [0.0] * 6)
    inspire_hand_max_joint_position: float = field(default_factory=lambda: [1000.0] * 6)
    inspire_hand_min_joint_effort: float = field(default_factory=lambda: [-500.0] * 6)
    inspire_hand_max_joint_effort: float = field(default_factory=lambda: [2000.0] * 6)