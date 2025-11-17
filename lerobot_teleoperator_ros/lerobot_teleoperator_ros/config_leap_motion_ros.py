from dataclasses import dataclass

from lerobot.teleoperators.config import TeleoperatorConfig


@TeleoperatorConfig.register_subclass("leap_motion_ros")
@dataclass
class LeapMotionROSTeleopConfig(TeleoperatorConfig):
    left_arm_topic: str = "/left_arm/teleop_cmd"
    right_arm_topic: str = "/right_arm/teleop_cmd"
