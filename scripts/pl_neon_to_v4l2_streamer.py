import cv2
import numpy as np
from pupil_labs.realtime_api.simple import discover_one_device


def main():
    # Look for devices. Returns as soon as it has found the first device.
    print("Looking for the next best device...")
    device = discover_one_device(max_search_duration_seconds=10)
    if device is None:
        print("No device found.")
        raise SystemExit(-1)

    print(f"Connecting to {device}...")

    fps = 30
    raw_width, raw_height = 1600, 1200  # native Neon resolution
    target_width, target_height = 320, 240 # desired output resolution
    crop_factor = 0.4 # crop to top-center 80% of the image
    crop_x1 = int(raw_width * crop_factor / 2)
    crop_x2 = int(raw_width * (1 - crop_factor / 2))
    crop_y1 = 0
    crop_y2 = int(raw_height * (1 - crop_factor))

    gst_str = ("appsrc ! videoconvert ! v4l2sink device=/dev/video20 sync=false")
    out = cv2.VideoWriter(gst_str, 
                        cv2.CAP_GSTREAMER,
                        cv2.VideoWriter_fourcc(*"MJPG"), 
                        fps, (target_width, target_height))

    if not out.isOpened():
        print("Failed to open VideoWriter.")
        raise SystemExit(-1)
    else: 
        print("VideoWriter opened successfully, writing to /dev/video20")

    cv2.setNumThreads(1)

    try:
        while True:
            # Get the next video frame and gaze data
            frame, gaze = device.receive_matched_scene_video_frame_and_gaze()
            
            # Draw gaze on the frame
            cv2.circle(
                frame.bgr_pixels,
                (int(gaze.x), int(gaze.y)),
                radius=80,
                color=(0, 0, 255),
                thickness=15,
            )

            # Crop to central region
            final_frame = frame.bgr_pixels[crop_y1:crop_y2, crop_x1:crop_x2]
            
            # Resize frame to target resolution
            final_frame = cv2.resize(final_frame, (target_width, target_height), interpolation=cv2.INTER_LINEAR)
            
            # Write to sink
            out.write(final_frame)
            
            if cv2.waitKey(1) & 0xFF == 27:
                break
    except KeyboardInterrupt:
        pass
    finally:
        print("Stopping...")
        device.close()  # explicitly stop auto-update


if __name__ == "__main__":
    main()