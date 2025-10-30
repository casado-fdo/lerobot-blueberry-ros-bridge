from typing import Any

from lerobot.teleoperators import Teleoperator

from .config_leap_motion_ros import LeapMotionROSTeleopConfig
from .leap_motion_ros_utils import LeapMotionROSInterface


class LeapMotionROSTeleop(Teleoperator):
    """
    Teleop class to use leap motion inputs (hand poses) to control robot arms (end effectors) and robot hands.
    """

    config_class = LeapMotionROSTeleopConfig
    name = "leap_motion_ros"

    def __init__(self, config: LeapMotionROSTeleopConfig):
        super().__init__(config)
        self.config = config
        self.robot_type = config.type

        self.leap: LeapMotionROSInterface | None = None

    @property
    def action_features(self) -> dict:        
        return {
            "dtype": "float32",
            "shape": (14,),
            "names": {
                "left_pos.x": 0,
                "left_pos.y": 1,
                "left_pos.z": 2,
                "left_rot.qx": 3,
                "left_rot.qy": 4,
                "left_rot.qz": 5,
                "left_rot.qw": 6,
                "right_pos.x": 7,
                "right_pos.y": 8,
                "right_pos.z": 9,
                "right_rot.qx": 10,
                "right_rot.qy": 11,
                "right_rot.qz": 12,
                "right_rot.qw": 13,
            },
        }

    @property
    def feedback_features(self) -> dict:
        return {}

    def connect(self) -> None:
        self.leap = LeapMotionROSInterface()
        self.leap.start()

    def get_action(self) -> dict[str, Any]:
        # Update the controller to get fresh inputs
        if self.leap is None:
            raise RuntimeError("Leap Motion is not connected. Please call connect() first.")

        # Get latest data from leap motion
        leap_action = self.leap.get_latest_data()

        action_dict = {
            "left_pos.x": leap_action[0],
            "left_pos.y": leap_action[1],
            "left_pos.z": leap_action[2],
            "left_rot.qx": leap_action[3],
            "left_rot.qy": leap_action[4],
            "left_rot.qz": leap_action[5],
            "left_rot.qw": leap_action[6],
            "right_pos.x": leap_action[7],
            "right_pos.y": leap_action[8],
            "right_pos.z": leap_action[9],
            "right_rot.qx": leap_action[10],
            "right_rot.qy": leap_action[11],
            "right_rot.qz": leap_action[12],
            "right_rot.qw": leap_action[13],
        }

        return action_dict

    def disconnect(self) -> None:
        """Disconnect from the leap motion."""
        if self.leap is not None:
            self.leap.stop()
            self.leap = None

    def is_connected(self) -> bool:
        """Check if leap motion is connected."""
        return self.leap is not None

    def calibrate(self) -> None:
        """Calibrate the leap motion."""
        # No calibration needed for leap motion
        pass

    def is_calibrated(self) -> bool:
        """Check if leap motion is calibrated."""
        # Leap motion doesn't require calibration
        return True

    def configure(self) -> None:
        """Configure the leap motion."""
        # No additional configuration needed
        pass

    def send_feedback(self, feedback: dict) -> None:
        """Send feedback to the leap motion."""
        # Leap motion doesn't support feedback
        pass
