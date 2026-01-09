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
            "shape": (24,),
            "names": {
                # Left arm: 6 DOF velocity commands
                "l_arm_linear.x": 0, "l_arm_linear.y": 1, "l_arm_linear.z": 2,
                "l_arm_angular.x": 3, "l_arm_angular.y": 4, "l_arm_angular.z": 5,
                # Left hand: 6 finger position commands
                "l_hand_pinky": 6, "l_hand_ring": 7, "l_hand_middle": 8, 
                "l_hand_index": 9, "l_hand_thumb1": 10, "l_hand_thumb2": 11,
                # Right arm: 6 DOF velocity commands
                "r_arm_linear.x": 12, "r_arm_linear.y": 13, "r_arm_linear.z": 14,
                "r_arm_angular.x": 15, "r_arm_angular.y": 16, "r_arm_angular.z": 17,
                # Right hand: 6 finger position commands
                "r_hand_pinky": 18, "r_hand_ring": 19, "r_hand_middle": 20, 
                "r_hand_index": 21, "r_hand_thumb1": 22, "r_hand_thumb2": 23,
            },
        }

    @property
    def feedback_features(self) -> dict:
        return {}

    def connect(self) -> None:
        self.leap = LeapMotionROSInterface(
            left_arm_topic=self.config.left_arm_topic,
            right_arm_topic=self.config.right_arm_topic,
            left_hand_topic=self.config.left_hand_topic,
            right_hand_topic=self.config.right_hand_topic,
        )

    def get_action(self) -> dict[str, Any]:
        # Update the controller to get fresh inputs
        if self.leap is None:
            raise RuntimeError("Leap Motion is not connected. Please call connect() first.")

        # Get latest data from leap motion
        leap_action = self.leap.get_latest_data()

        action_dict = {
            # Left arm
            "l_arm_linear.x": leap_action[0], "l_arm_linear.y": leap_action[1], "l_arm_linear.z": leap_action[2],
            "l_arm_angular.x": leap_action[3], "l_arm_angular.y": leap_action[4], "l_arm_angular.z": leap_action[5],
            # Left hand
            'l_hand_pinky': leap_action[6], 'l_hand_ring': leap_action[7], 'l_hand_middle': leap_action[8], 
            'l_hand_index': leap_action[9], 'l_hand_thumb1': leap_action[10], 'l_hand_thumb2': leap_action[11],
            # Right arm
            "r_arm_linear.x": leap_action[12], "r_arm_linear.y": leap_action[13], "r_arm_linear.z": leap_action[14],
            "r_arm_angular.x": leap_action[15], "r_arm_angular.y": leap_action[16], "r_arm_angular.z": leap_action[17],
            # Right hand
            'r_hand_pinky': leap_action[18], 'r_hand_ring': leap_action[19], 'r_hand_middle': leap_action[20], 
            'r_hand_index': leap_action[21], 'r_hand_thumb1': leap_action[22], 'r_hand_thumb2': leap_action[23],
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
