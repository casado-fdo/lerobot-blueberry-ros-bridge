from dataclasses import dataclass
from pathlib import Path

from lerobot.cameras.configs import CameraConfig, ColorMode, Cv2Backends, Cv2Rotation

__all__ = ["PLNeonCameraConfig", "ColorMode", "Cv2Rotation", "Cv2Backends"]


@CameraConfig.register_subclass("plneon")
@dataclass
class PLNeonCameraConfig(CameraConfig):
    """Configuration class for OpenCV-based streaming from the Pupil Labs Neon device.


    Attributes:
        index_or_path: A Path object pointing to a video file.
        color_mode: Color mode for image output (RGB or BGR). Defaults to RGB.
        warmup_s: Time reading frames before returning from connect (in seconds)

    Note:
        - Only 3-channel color output (RGB/BGR) is currently supported.
        - Setting FOURCC can help achieve higher frame rates on some cameras.
    """

    index_or_path: int | Path
    color_mode: ColorMode = ColorMode.RGB
    backend: Cv2Backends = Cv2Backends.ANY
    warmup_s: int = 1

    def __post_init__(self) -> None:
        if self.color_mode not in (ColorMode.RGB, ColorMode.BGR):
            raise ValueError(
                f"`color_mode` is expected to be {ColorMode.RGB.value} or {ColorMode.BGR.value}, but {self.color_mode} is provided."
            )