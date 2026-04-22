from dataclasses import dataclass

from lerobot.teleoperators.config import TeleoperatorConfig


@TeleoperatorConfig.register_subclass("blueberry_teleop")
@dataclass
class BlueberryTeleopConfig(TeleoperatorConfig):
    left_arm_topic: str = "/left_arm/teleop_cmd"
    right_arm_topic: str = "/right_arm/teleop_cmd"
    left_hand_topic: str = "/left_hand/teleop_cmd"
    right_hand_topic: str = "/right_hand/teleop_cmd"
    base_topic: str = "/base/teleop_cmd"
    device_timeout_sec: float = 1.0