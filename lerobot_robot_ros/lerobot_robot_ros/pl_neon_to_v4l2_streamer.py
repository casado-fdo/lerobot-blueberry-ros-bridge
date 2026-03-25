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
    fps,
    target_width,
    target_height,
    v4l2_device0,
    v4l2_device1 = None,
    crop_keep_ratio=0.4,        # How much of the image to keep (0-1)
    vertical_offset_ratio=0.2,  # Vertical bias (0 = very top, 0.5 = centre)
    search_timeout=10,
):
    pl_device = None
    out0 = None
    out1 = None

    try:
        logger.info("NeonV4L2Process: Looking for the next best Neon device...")
        pl_device = discover_one_device(
            max_search_duration_seconds=search_timeout
        )

        if pl_device is None:
            raise RuntimeError("No Neon device found.")

        logger.info(f"NeonV4L2Process: Connecting to {pl_device}...")

        # Compute crop coordinates
        crop_width = int(RAW_WIDTH * crop_keep_ratio)
        crop_height = int(RAW_HEIGHT * crop_keep_ratio)
        crop_x1 = (RAW_WIDTH - crop_width) // 2
        crop_x2 = crop_x1 + crop_width
        crop_y1 = int(RAW_HEIGHT * vertical_offset_ratio)
        crop_y2 = crop_y1 + crop_height
        crop_y2 = min(crop_y2, RAW_HEIGHT) # Safety clamp (just in case)


        gst_str0 = (f"appsrc ! videoconvert ! v4l2sink device={v4l2_device0} sync=false")
        out0 = cv2.VideoWriter(gst_str0, 
                            cv2.CAP_GSTREAMER,
                            cv2.VideoWriter_fourcc(*"MJPG"), 
                            fps, (target_width, target_height))
        if v4l2_device1 is not None:
            gst_str1 = (f"appsrc ! videoconvert ! v4l2sink device={v4l2_device1} sync=false")
            out1 = cv2.VideoWriter(gst_str1,
                                cv2.CAP_GSTREAMER,
                                cv2.VideoWriter_fourcc(*"MJPG"), 
                                fps, (target_width, target_height))

        if not out0.isOpened() or (v4l2_device1 is not None and not out1.isOpened()):
            raise RuntimeError("Failed to open VideoWriter.")

        if v4l2_device1 is not None:
            logger.info(f"NeonV4L2Process: VideoWriter opened successfully, writing to {v4l2_device0} and {v4l2_device1}")
        else:
            logger.info(f"NeonV4L2Process: VideoWriter opened successfully, writing to {v4l2_device0}")

        cv2.setNumThreads(1)

        while not stop_event.is_set():
            
            # Get the next video frame and gaze data
            frame, gaze = pl_device.receive_matched_scene_video_frame_and_gaze()

            if v4l2_device1 is not None:
                # Create a copy of the frame, resize it and write it to the second V4L2 device (raw data)
                frame_raw = frame.bgr_pixels.copy()
                final_frame_raw = cv2.resize(frame_raw, (target_width, target_height), interpolation=cv2.INTER_LINEAR)
                out1.write(final_frame_raw)
            
            # Draw gaze on the frame
            cv2.circle(
                frame.bgr_pixels,
                (int(gaze.x), int(gaze.y)),
                radius=45,
                color=(0, 0, 255),
                thickness=8,
            )

            # Crop to central region
            final_frame_gaze = frame.bgr_pixels[crop_y1:crop_y2, crop_x1:crop_x2]
            
            # Resize frame to target resolution
            final_frame_gaze = cv2.resize(final_frame_gaze, (target_width, target_height), interpolation=cv2.INTER_LINEAR)
            
            # Write to the first V4L2 device (gaze-annotated)
            out0.write(final_frame_gaze)
                
    except Exception as e:
        logger.error(f"NeonV4L2Process: Error: {e}")

    finally:
        logger.info("NeonV4L2Process: Shutting down...")

        if out0 is not None:
            out0.release()

        if out1 is not None:
            out1.release()

        if pl_device is not None:
            pl_device.close()


class NeonV4L2Process:
    def __init__(
        self,
        v4l2_device0="/dev/video20",
        v4l2_device1=None,
        fps=15,
        target_width=320,
        target_height=240,
        crop_keep_ratio=0.45,
        vertical_offset_ratio=0.25,
        search_timeout=10,
    ):
        self.v4l2_device0 = v4l2_device0
        self.v4l2_device1 = v4l2_device1
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
                self.fps,
                self.target_width,
                self.target_height,
                self.v4l2_device0,
                self.v4l2_device1,
                self.crop_keep_ratio,
                self.vertical_offset_ratio,
                self.search_timeout,
            ),
            daemon=True,
        )

        self._process.start()
        time.sleep(7.0) # Wait a few seconds to let the stream start
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
