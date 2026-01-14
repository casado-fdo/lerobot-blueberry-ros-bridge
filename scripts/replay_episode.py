# !/usr/bin/env python

import time

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.model.kinematics import RobotKinematics
from lerobot.processor import RobotAction, RobotObservation, RobotProcessorPipeline
from lerobot.processor.converters import (
    robot_action_observation_to_transition,
    transition_to_robot_action,
)
from lerobot.utils.constants import ACTION
from lerobot.utils.robot_utils import precise_sleep
from utils import log_say
import os
import argparse
from lerobot_robot_ros import BlueberryROS, BlueberryROSConfig

HF_USERNAME = os.getenv("HUGGINGFACE_USERNAME", "your-username-here")

def main(hf_dataset_name: str, episode_idx: int):
    log_say("Initialising episode replay...", play_sounds=False)


    HF_REPO_ID = f"{HF_USERNAME}/{hf_dataset_name}"

    # Initialize the robot config
    robot_config = BlueberryROSConfig() # default config
        
    # Initialize the robot
    robot = BlueberryROS(robot_config)

    # Fetch the dataset to replay
    dataset = LeRobotDataset(HF_REPO_ID, episodes=[episode_idx])
    # Filter dataset to only include frames from the specified episode since episodes are chunked in dataset V3.0
    episode_frames = dataset.hf_dataset.filter(lambda x: x["episode_index"] == episode_idx)
    actions = episode_frames.select_columns(ACTION)

    # Connect to the robot
    robot.connect()

    if not robot.is_connected:
        raise ValueError("Robot is not connected!")

    log_say(f"Replaying episode {episode_idx}", play_sounds=False)
    for idx in range(len(episode_frames)):
        t0 = time.perf_counter()

        # Get recorded action from dataset
        robot_action = {
            name: float(actions[idx][ACTION][i]) for i, name in enumerate(dataset.features[ACTION]["names"])
        }

        # Get robot observation
        robot_obs = robot.get_observation()

        # Send action to robot
        _ = robot.send_action(robot_action)

        precise_sleep(max(1.0 / dataset.fps - (time.perf_counter() - t0), 0.0))

    # Clean up
    robot.disconnect()


if __name__ == "__main__":    
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_id", type=str, required=True, help="HuggingFace dataset name to replay from")
    parser.add_argument("--episode_idx", type=int, required=False, help="Episode index to replay", default=0)
    args = parser.parse_args()
    main(args.dataset_id, args.episode_idx)