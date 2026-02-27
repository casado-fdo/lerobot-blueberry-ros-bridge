from typing import Any

from lerobot.teleoperators import Teleoperator

from .config_teleop_blueberry import BlueberryTeleopConfig
from .ros_interface_teleop_blueberry import BlueberryTeleopROSInterface


class BlueberryTeleop(Teleoperator):
    """
    Teleop class to teleoperate Blueberry robotic wheelchair. 
    It uses Leap Motion inputs (hand poses) to control robot arms (end effectors) and robot hands.
    It also includes pedal inputs (Joy commands) for simultaneous control of the mobile base.
    """

    config_class = BlueberryTeleopConfig
    name = "blueberry_teleop"

    def __init__(self, config: BlueberryTeleopConfig):
        super().__init__(config)
        self.config = config
        self.robot_type = config.type

        self.ros_interface: BlueberryTeleopROSInterface | None = None

    @property
    def action_features(self) -> dict:        
        return {
            "dtype": "float32",
            "shape": (26,),
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
                # Base: 2 joystick commands
                "base_joy.x": 24, "base_joy.y": 25,
            },
        }

    @property
    def feedback_features(self) -> dict:
        return {}

    def connect(self) -> None:
        self.ros_interface = BlueberryTeleopROSInterface(
            left_arm_topic=self.config.left_arm_topic,
            right_arm_topic=self.config.right_arm_topic,
            left_hand_topic=self.config.left_hand_topic,
            right_hand_topic=self.config.right_hand_topic,
            base_topic=self.config.base_topic,
        )

    def get_action(self) -> dict[str, Any]:
        # Update the controller to get fresh inputs
        if self.ros_interface is None:
            raise RuntimeError("Blueberry ROS interface is not connected. Please call connect() first.")

        # Get latest data from leap motion and pedals
        last_action = self.ros_interface.get_latest_data()

        action_dict = {
            # Left arm
            "l_arm_linear.x": last_action[0], "l_arm_linear.y": last_action[1], "l_arm_linear.z": last_action[2],
            "l_arm_angular.x": last_action[3], "l_arm_angular.y": last_action[4], "l_arm_angular.z": last_action[5],
            # Left hand
            'l_hand_pinky': last_action[6], 'l_hand_ring': last_action[7], 'l_hand_middle': last_action[8], 
            'l_hand_index': last_action[9], 'l_hand_thumb1': last_action[10], 'l_hand_thumb2': last_action[11],
            # Right arm
            "r_arm_linear.x": last_action[12], "r_arm_linear.y": last_action[13], "r_arm_linear.z": last_action[14],
            "r_arm_angular.x": last_action[15], "r_arm_angular.y": last_action[16], "r_arm_angular.z": last_action[17],
            # Right hand
            'r_hand_pinky': last_action[18], 'r_hand_ring': last_action[19], 'r_hand_middle': last_action[20], 
            'r_hand_index': last_action[21], 'r_hand_thumb1': last_action[22], 'r_hand_thumb2': last_action[23],
            # Base
            'base_joy.x': last_action[24], 'base_joy.y': last_action[25],
        }

        return action_dict

    def disconnect(self) -> None:
        """Disconnect from the Blueberry ROS teleoperation interface."""
        if self.ros_interface is not None:
            self.ros_interface.stop()
            self.ros_interface = None

    def is_connected(self) -> bool:
        """Check if Blueberry ROS teleoperation interface is connected."""
        return self.ros_interface is not None

    def calibrate(self) -> None:
        # No calibration needed for Blueberry ROS teleoperation interface
        pass

    def is_calibrated(self) -> bool:
        # Blueberry teleoperation doesn't require calibration
        return True

    def configure(self) -> None:
        # No additional configuration needed
        pass

    def send_feedback(self, feedback: dict) -> None:
        # Blueberry teleoperation doesn't support feedback
        pass
