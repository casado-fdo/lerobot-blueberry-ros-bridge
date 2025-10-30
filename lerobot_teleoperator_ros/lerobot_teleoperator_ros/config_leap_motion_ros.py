from dataclasses import dataclass

from lerobot.teleoperators.config import TeleoperatorConfig


@TeleoperatorConfig.register_subclass("leap_motion_ros")
@dataclass
class LeapMotionROSTeleopConfig(TeleoperatorConfig):
    use_gripper: bool = False
