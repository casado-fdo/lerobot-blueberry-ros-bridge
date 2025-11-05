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
                "left_linear.x": 0,
                "left_linear.y": 1,
                "left_linear.z": 2,
                "left_angular.x": 3,
                "left_angular.y": 4,
                "left_angular.z": 5,
                "right_linear.x": 6,
                "right_linear.y": 7,
                "right_linear.z": 8,
                "right_angular.x": 9,
                "right_angular.y": 10,
                "right_angular.z": 11,
            },
        }

    @property
    def feedback_features(self) -> dict:
        return {}

    def connect(self) -> None:
        self.leap = LeapMotionROSInterface()

    def get_action(self) -> dict[str, Any]:
        # Update the controller to get fresh inputs
        if self.leap is None:
            raise RuntimeError("Leap Motion is not connected. Please call connect() first.")

        # Get latest data from leap motion
        leap_action = self.leap.get_latest_data()

        action_dict = {
            "left_linear.x": leap_action[0],
            "left_linear.y": leap_action[1],
            "left_linear.z": leap_action[2],
            "left_angular.x": leap_action[3],
            "left_angular.y": leap_action[4],
            "left_angular.z": leap_action[5],
            "right_linear.x": leap_action[6],
            "right_linear.y": leap_action[7],
            "right_linear.z": leap_action[8],
            "right_angular.x": leap_action[9],
            "right_angular.y": leap_action[10],
            "right_angular.z": leap_action[11],
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
