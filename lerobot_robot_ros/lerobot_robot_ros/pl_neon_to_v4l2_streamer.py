import cv2
import time
import multiprocessing as mp
from pupil_labs.realtime_api.simple import discover_one_device
import logging

RAW_WIDTH = 1600
RAW_HEIGHT = 1200
logger = logging.getLogger(__name__)

def _neon_stream_loop(
    stop_event,
    v4l2_device,
    fps,
    target_width,
    target_height,
    crop_keep_ratio,
    vertical_offset_ratio,
    search_timeout,
):
    device = None
    out = None

    try:
        logger.info("NeonV4L2Process: Looking for the next best Neon device...")
        device = discover_one_device(
            max_search_duration_seconds=search_timeout
        )

        if device is None:
            raise RuntimeError("No Neon device found.")

        logger.info(f"NeonV4L2Process: Connecting to {device}...")

        crop_keep_ratio = 0.4 # How much of the image to keep (0-1)
        vertical_offset_ratio = 0.2 # Vertical bias (0 = very top, 0.5 = centre)

        # Compute crop coordinates
        crop_width = int(RAW_WIDTH * crop_keep_ratio)
        crop_height = int(RAW_HEIGHT * crop_keep_ratio)
        crop_x1 = (RAW_WIDTH - crop_width) // 2
        crop_x2 = crop_x1 + crop_width
        crop_y1 = int(RAW_HEIGHT * vertical_offset_ratio)
        crop_y2 = crop_y1 + crop_height
        crop_y2 = min(crop_y2, RAW_HEIGHT) # Safety clamp (just in case)


        gst_str = (f"appsrc ! videoconvert ! v4l2sink device={v4l2_device} sync=false")
        out = cv2.VideoWriter(gst_str, 
                            cv2.CAP_GSTREAMER,
                            cv2.VideoWriter_fourcc(*"MJPG"), 
                            fps, (target_width, target_height))

        if not out.isOpened():
            raise RuntimeError("Failed to open VideoWriter.")

        logger.info(f"NeonV4L2Process: VideoWriter opened successfully, writing to {v4l2_device}")

        cv2.setNumThreads(1)

        while not stop_event.is_set():
            
            # Get the next video frame and gaze data
            frame, gaze = device.receive_matched_scene_video_frame_and_gaze()
            
            # Draw gaze on the frame
            cv2.circle(
                frame.bgr_pixels,
                (int(gaze.x), int(gaze.y)),
                radius=45,
                color=(0, 0, 255),
                thickness=8,
            )

            # Crop to central region
            final_frame = frame.bgr_pixels[crop_y1:crop_y2, crop_x1:crop_x2]
            
            # Resize frame to target resolution
            final_frame = cv2.resize(final_frame, (target_width, target_height), interpolation=cv2.INTER_LINEAR)
            
            # Write to sink
            out.write(final_frame)
                
    except Exception as e:
        logger.error(f"NeonV4L2Process: Error: {e}")

    finally:
        logger.info("NeonV4L2Process: Shutting down...")

        if out is not None:
            out.release()

        if device is not None:
            device.close()


class NeonV4L2Process:
    def __init__(
        self,
        v4l2_device="/dev/video20",
        fps=15,
        target_width=320,
        target_height=240,
        crop_keep_ratio=0.4,
        vertical_offset_ratio=0.2,
        search_timeout=10,
    ):
        self.v4l2_device = v4l2_device
        self.fps = fps
        self.target_width = target_width
        self.target_height = target_height
        self.crop_keep_ratio = crop_keep_ratio
        self.vertical_offset_ratio = vertical_offset_ratio
        self.search_timeout = search_timeout

        self._stop_event = mp.Event()
        self._process = None

    def start(self):
        if self._process is not None and self._process.is_alive():
            return

        self._stop_event.clear()

        self._process = mp.Process(
            target=_neon_stream_loop,
            args=(
                self._stop_event,
                self.v4l2_device,
                self.fps,
                self.target_width,
                self.target_height,
                self.crop_keep_ratio,
                self.vertical_offset_ratio,
                self.search_timeout,
            ),
            daemon=True,
        )

        self._process.start()
        time.sleep(6.0) # Wait a few seconds to let the stream start
        logger.info("Neon streaming process started.")

    def stop(self, timeout=5):
        if self._process is None:
            return

        logger.info("Stopping Neon streaming process...")
        self._stop_event.set()
        self._process.join(timeout)

        if self._process.is_alive():
            logger.info("Force terminating Neon process.")
            self._process.terminate()
            self._process.join()

        self._process = None
