import time
from functools import cached_property
from typing import Any

from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.robots import Robot
from lerobot.robots.utils import ensure_safe_goal_position
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from .config import ROSConfig
from .ros_interface import ROSInterface



class ROSRobot(Robot):
    config_class = ROSConfig
    name = "ros_robot"

    def __init__(self, config: ROSConfig):
        super().__init__(config)
        self.config = config
        self.ros_interface = ROSInterface(config.ros_interface)
        #self.cameras = make_cameras_from_configs(config.cameras)

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        return {
            # cam: (self.config.cameras[cam].height, self.config.cameras[cam].width, 3) for cam in self.cameras
        }

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        all_joint_names = self.config.ros_interface.arm_joint_names.copy()
        motor_state_ft = {f"{motor}.pos": float for motor in all_joint_names}
        return {**motor_state_ft, **self._cameras_ft}
        return None

    @cached_property
    def action_features(self) -> dict[str, type]:
        return {
            "left_linear.x": float,
            "left_linear.y": float,
            "left_linear.z": float,
            "left_angular.x": float,
            "left_angular.y": float,
            "left_angular.z": float,
            "right_linear.x": float,
            "right_linear.y": float,
            "right_linear.z": float,
            "right_angular.x": float,
            "right_angular.y": float,
            "right_angular.z": float,
        }

    @property
    def is_connected(self) -> bool:
        return self.ros_interface.is_connected # and all(cam.is_connected for cam in self.cameras.values())

    def connect(self, calibrate: bool = True) -> None:
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")

        #for cam in self.cameras.values():
        #    cam.connect()
        self.ros_interface.connect()

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

        obs_dict: dict[str, Any] = {}
        joint_state = self.ros_interface.joint_state
        if joint_state is None:
            raise ValueError("Joint state is not available yet.")
        obs_dict.update({f"{joint}.pos": pos for joint, pos in joint_state["position"].items()})

        # Capture images from cameras
        #for cam_key, cam in self.cameras.items():
        #    start = time.perf_counter()
        #    try:
        #        obs_dict[cam_key] = cam.async_read(timeout_ms=300)
        #    except Exception as e:
        #        obs_dict[cam_key] = None
        #    dt_ms = (time.perf_counter() - start) * 1e3

        return obs_dict

    def send_action(self, action: dict[str, float]) -> dict[str, float]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        #self.ros_interface.send_joint_position_command(joint_positions)
        self.ros_interface.send_leap_command(action)

        return action

    def disconnect(self):
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        #for cam in self.cameras.values():
        #    cam.disconnect()
        self.ros_interface.disconnect()


class BlueberryROS(ROSRobot):
    pass