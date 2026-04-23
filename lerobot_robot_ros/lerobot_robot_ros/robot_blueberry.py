import time
import logging
from functools import cached_property
from typing import Any

from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.robots import Robot
from lerobot.robots.utils import ensure_safe_goal_position
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from .config_blueberry import BlueberryROSConfig
from .ros_interface_blueberry import BlueberryROSInterface
from .pl_neon_to_v4l2_streamer import NeonV4L2Process

logger = logging.getLogger(__name__)

class BlueberryROS(Robot):
    config_class = BlueberryROSConfig
    name = "blueberry_ros"

    def __init__(self, config: BlueberryROSConfig):
        super().__init__(config)
        self.config = config
        self.ros_interface = BlueberryROSInterface(config)
        
        # Get user camera config if available, otherwise use gaze camera config as fallback
        if "user" in self.config.cameras:
            pl_neon_cfg = self.config.cameras["user"]
        elif "user_gaze" in self.config.cameras:
            pl_neon_cfg = self.config.cameras["user_gaze"]
        else:
            # Neither user camera is available, use default values
            pl_neon_cfg = type('Config', (), {
                'fps': config.cam_fps if hasattr(config, 'cam_fps') else 15,
                'width': config.cam_width if hasattr(config, 'cam_width') else 320,
                'height': config.cam_height if hasattr(config, 'cam_height') else 240
            })()
        
        self.pl_neon_streamer = NeonV4L2Process(v4l2_device0="/dev/video20" if config.record_user_cam else None,  
                                                v4l2_device1="/dev/video21" if config.record_user_gaze_cam else None,
                                                fps=pl_neon_cfg.fps, 
                                                target_width=pl_neon_cfg.width, 
                                                target_height=pl_neon_cfg.height)
        self.cameras = make_cameras_from_configs(config.cameras)

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        return {
            cam: (self.config.cameras[cam].height, self.config.cameras[cam].width, 3) for cam in self.cameras
        }

    @property
    def _robot_state_ft(self) -> dict[str, type]:
        return {
            f"{motor}.pos": float for motor in self.config.blueberry_joint_names
        } | {
            f"{motor}.effort": float for motor in self.config.blueberry_joint_names
        }    

    @property
    def _gaze_features_ft(self) -> dict[str, type]:
        return {
            "gaze.x": float,
            "gaze.y": float,
            "gaze.valid": float,
        }

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        return {**self._robot_state_ft, **self._cameras_ft, **self._gaze_features_ft}
 
    @cached_property
    def action_features(self) -> dict[str, type]:
        return {
            # Left arm: 6 DOF velocity commands
            "l_arm_linear.x": float, "l_arm_linear.y": float, "l_arm_linear.z": float,
            "l_arm_angular.x": float, "l_arm_angular.y": float, "l_arm_angular.z": float,
            # Left hand: 6 finger position commands
            "l_hand_pinky": float, "l_hand_ring": float, "l_hand_middle": float, 
            "l_hand_index": float, "l_hand_thumb1": float, "l_hand_thumb2": float,
            # Right arm: 6 DOF velocity commands
            "r_arm_linear.x": float, "r_arm_linear.y": float, "r_arm_linear.z": float,
            "r_arm_angular.x": float, "r_arm_angular.y": float, "r_arm_angular.z": float,
            # Right hand: 6 finger position commands
            "r_hand_pinky": float, "r_hand_ring": float, "r_hand_middle": float, 
            "r_hand_index": float, "r_hand_thumb1": float, "r_hand_thumb2": float,
            # Base: 2 DOF joy commands
            "base_joy.x": float, "base_joy.y": float,
        }

    @property
    def is_connected(self) -> bool:
        return self.ros_interface.is_connected and all(cam.is_connected for cam in self.cameras.values())

    def connect(self, calibrate: bool = True) -> None:
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")
        self.pl_neon_streamer.start()
        for cam in self.cameras.values():
           cam.connect()
        self.ros_interface.connect()
        logger.info(f"{self} connected")

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        pass  # robot must be calibrated before running LeRobot

    def configure(self) -> None:
        pass  # robot must be configured before running LeRobot

    def get_observation(self) -> dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")
        start = time.perf_counter()
        obs_dict: dict[str, Any] = {}
        joint_state = self.ros_interface.joint_state
        if joint_state is None:
            raise ValueError("Joint state is not available yet.")
        obs_dict.update({f"{joint}.pos": pos for joint, pos in joint_state["position"].items()})
        obs_dict.update({f"{joint}.effort": effort for joint, effort in joint_state["effort"].items()})
        dt_ms = (time.perf_counter() - start) * 1e3
        logger.debug(f"{self}: joint state read in {dt_ms:.2f} ms")

        # Capture images from cameras
        for cam_key, cam in self.cameras.items():
            start = time.perf_counter()
            try:
                obs_dict[cam_key] = cam.async_read(timeout_ms=300)
            except Exception as e:
                obs_dict[cam_key] = None
            dt_ms = (time.perf_counter() - start) * 1e3
            logger.debug(f"{self}: {cam_key} read in {dt_ms:.2f} ms")

        # Get latest gaze coordinates and validity mask
        start = time.perf_counter()
        try:
            gaze_x, gaze_y, gaze_valid = self.pl_neon_streamer.get_latest_gaze()
            obs_dict["gaze.x"] = gaze_x
            obs_dict["gaze.y"] = gaze_y
            obs_dict["gaze.valid"] = gaze_valid
        except Exception as e:
            obs_dict["gaze.x"] = 0.0
            obs_dict["gaze.y"] = 0.0
            obs_dict["gaze.valid"] = 0
            logger.debug(f"{self}: Failed to get gaze data: {e}")
        dt_ms = (time.perf_counter() - start) * 1e3
        logger.debug(f"{self}: gaze data read in {dt_ms:.2f} ms")

        return obs_dict

    def send_action(self, action: dict[str, float]) -> dict[str, float]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        self.ros_interface.send_action_command(action)

        return action

    def disconnect(self):
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        for cam in self.cameras.values():
            cam.disconnect()
        self.ros_interface.disconnect()
        self.pl_neon_streamer.stop()
        logger.info(f"{self} disconnected")