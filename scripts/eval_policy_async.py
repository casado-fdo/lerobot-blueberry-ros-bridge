# !/usr/bin/env python

import threading
from lerobot.async_inference.configs import RobotClientConfig
from lerobot.async_inference.helpers import visualize_action_queue_size
from lerobot_robot_ros import BlueberryROSConfig
from lerobot_robot_ros.async_robot_client import RobotClient
import os
import argparse

NUM_EPISODES = int(os.getenv("RECORDING_NUM_EPISODES", "1"))
FPS = int(os.getenv("RECORDING_FPS", "30"))
EPISODE_TIME_SEC = int(os.getenv("RECORDING_EPISODE_TIME_SEC", "10"))
RESET_TIME_SEC = int(os.getenv("RECORDING_RESET_TIME_SEC", "5"))
TASK_DESCRIPTION = os.getenv("RECORDING_TASK_DESCRIPTION", "No task description provided.")
PLAY_SOUNDS = bool(os.getenv("RECORDING_PLAY_SOUNDS", "True").lower() in ("true", "1", "yes"))
HF_USERNAME = os.getenv("HUGGINGFACE_USERNAME", "your-username-here")
SERVER_ADDRESS = "127.0.0.1:8090"

def main(hf_policy_id: str, policy_type: str = "act"):
    #log_say("Initialising policy evaluation...", play_sounds=PLAY_SOUNDS)

    hf_policy_repo_id = f"{HF_USERNAME}/{hf_policy_id}"

    # Initialize the robot config
    robot_config = BlueberryROSConfig() # default config

    # Create client configuration
    client_cfg = RobotClientConfig(
        robot=robot_config,
        server_address=SERVER_ADDRESS,
        policy_device="cuda",
        client_device="cpu",
        policy_type=policy_type,
        pretrained_name_or_path=hf_policy_repo_id,
        chunk_size_threshold=0.7,  # g
        actions_per_chunk=50,  # make sure this is less than the max actions of the policy
    )

    # Create and start client
    client = RobotClient(client_cfg)

    # Provide a textual description of the task
    task = TASK_DESCRIPTION

    if client.start():
        # Start action receiver thread
        action_receiver_thread = threading.Thread(target=client.receive_actions, daemon=True)
        action_receiver_thread.start()

        try:
            # Run the control loop
            client.control_loop(task)
        except KeyboardInterrupt:
            client.stop()
            action_receiver_thread.join()
            # (Optionally) plot the action queue size
            visualize_action_queue_size(client.action_queue_size)



if __name__ == "__main__":    
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy_id", type=str, required=True, help="HuggingFace policy name to evaluate")
    parser.add_argument("--policy_type", type=str, default="act", help="Type of policy to evaluate (default: act)")
    args = parser.parse_args()
    main(args.policy_id, args.policy_type)