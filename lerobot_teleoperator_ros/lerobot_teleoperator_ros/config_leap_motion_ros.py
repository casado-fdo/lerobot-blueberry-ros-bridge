from dataclasses import dataclass

from lerobot.teleoperators.config import TeleoperatorConfig


@TeleoperatorConfig.register_subclass("leap_motion_ros")
@dataclass
class LeapMotionROSTeleopConfig(TeleoperatorConfig):
    left_arm_topic: str = "/left_arm/teleop_cmd"
    right_arm_topic: str = "/right_arm/teleop_cmd"
    left_hand_topic: str = "/left_hand/teleop_cmd"
    right_hand_topic: str = "/right_hand/teleop_cmd"