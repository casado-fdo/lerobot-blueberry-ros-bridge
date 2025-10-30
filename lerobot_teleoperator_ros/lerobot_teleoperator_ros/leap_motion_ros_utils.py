import numpy as np
import logging

class LeapMotionROSInterface():
    """Stream hand pose commands (left and right) from leap motion input."""

    def __init__(self):
        # TODO: Initialize ROS node and subscribers here
        pass

    def start(self):
        # TODO: Check here that the ros node is running
        logging.info("Starting Leap Motion ROS Interface...")
        pass

    def stop(self):
        #TODO: Clean up any resources if needed
        logging.info("Stopping Leap Motion ROS Interface...")
        pass

    def get_latest_data(self) -> np.ndarray:
        # TODO: Implement ROS topic subscription to get latest Leap Motion data
        return np.zeros(14, dtype=np.float32)
