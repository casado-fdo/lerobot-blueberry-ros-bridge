import cv2
import os
import time
from pupil_labs.realtime_api.simple import discover_one_device


def main():
    # Look for devices. Returns as soon as it has found the first device.
    print("Looking for the next best device...")
    device = discover_one_device(max_search_duration_seconds=10)
    if device is None:
        print("No device found.")
        raise SystemExit(-1)

    print(f"Connecting to {device}...")

    fps = int(os.getenv("RECORDING_FPS", "15"))
    raw_width, raw_height = 1600, 1200  # native Neon resolution
    target_width = int(os.getenv("RECORDING_VIDEO_WIDTH", "320"))
    target_height = int(os.getenv("RECORDING_VIDEO_HEIGHT", "240"))
    crop_keep_ratio = 0.4 # How much of the image to keep (0-1)
    vertical_offset_ratio = 0.2 # Vertical bias (0 = very top, 0.5 = centre)

    # Compute crop coordinates
    crop_width = int(raw_width * crop_keep_ratio)
    crop_height = int(raw_height * crop_keep_ratio)
    crop_x1 = (raw_width - crop_width) // 2
    crop_x2 = crop_x1 + crop_width
    crop_y1 = int(raw_height * vertical_offset_ratio)
    crop_y2 = crop_y1 + crop_height
    crop_y2 = min(crop_y2, raw_height) # Safety clamp (just in case)


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

    frame_interval = 1.0 / fps  # Time between frames in seconds
    last_frame_time = time.time()

    try:
        while True:
            current_time = time.time()
            
            # Only process frame if enough time has passed
            if current_time - last_frame_time >= frame_interval:
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
                
                last_frame_time = current_time
            
            if cv2.waitKey(1) & 0xFF == 27:
                break
    except KeyboardInterrupt:
        pass
    finally:
        print("Stopping...")
        device.close()  # explicitly stop auto-update


if __name__ == "__main__":
    main()