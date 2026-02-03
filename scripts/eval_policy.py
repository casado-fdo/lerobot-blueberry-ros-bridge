# !/usr/bin/env python

import time

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.processor import RobotAction, RobotObservation, RobotProcessorPipeline, make_default_processors
from lerobot.utils.constants import ACTION
from lerobot.utils.robot_utils import precise_sleep
from lerobot.policies.act.modeling_act import ACTPolicy, ACTTemporalEnsembler
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies.xvla.modeling_xvla import XVLAPolicy
from lerobot.policies.factory import make_pre_post_processors
from lerobot_robot_ros import BlueberryROS, BlueberryROSConfig
from lerobot.utils.control_utils import init_keyboard_listener
from lerobot.scripts.lerobot_record import record_loop
from utils import log_say
import os
import argparse

NUM_EPISODES = int(os.getenv("RECORDING_NUM_EPISODES", "1"))
FPS = int(os.getenv("RECORDING_FPS", "30"))
EPISODE_TIME_SEC = int(os.getenv("RECORDING_EPISODE_TIME_SEC", "10"))
RESET_TIME_SEC = int(os.getenv("RECORDING_RESET_TIME_SEC", "5"))
TASK_DESCRIPTION = os.getenv("RECORDING_TASK_DESCRIPTION", "No task description provided.")
PLAY_SOUNDS = bool(os.getenv("RECORDING_PLAY_SOUNDS", "True").lower() in ("true", "1", "yes"))
HF_USERNAME = os.getenv("HUGGINGFACE_USERNAME", "your-username-here")
#HF_DATASET_NAME = os.getenv("HUGGINGFACE_DATASET_NAME", "default")

def main(hf_policy_id: str, hf_dataset_id: str = None, policy_type: str = "act"):
    log_say("Initialising policy evaluation...", play_sounds=PLAY_SOUNDS)

    hf_policy_repo_id = f"{HF_USERNAME}/{hf_policy_id}"
    hf_dataset_repo_id = f"{HF_USERNAME}/{hf_dataset_id}"

    # Initialize the robot config
    robot_config = BlueberryROSConfig() # default config
        
    # Initialize the robot
    robot = BlueberryROS(robot_config)

    teleop_action_processor, robot_action_processor, robot_observation_processor = make_default_processors()


    # Create policy
    if policy_type.lower() == "act":
        policy = ACTPolicy.from_pretrained(hf_policy_repo_id)
    elif policy_type.lower() == "smolvla":
        policy = SmolVLAPolicy.from_pretrained(hf_policy_repo_id)
    elif policy_type.lower() == "xvla":
        policy = XVLAPolicy.from_pretrained(hf_policy_repo_id)
    else:
        raise ValueError(f"Unsupported policy type: {policy_type}")

    # Fetch the dataset for stats
    dataset = LeRobotDataset(hf_dataset_repo_id)

    # Change some parameters in the policy config
    policy.config.temporal_ensemble_coeff=0.01
    policy.temporal_ensembler=ACTTemporalEnsembler(policy.config.temporal_ensemble_coeff, policy.config.chunk_size)
    policy.config.n_action_steps=1 #min(policy.config.chunk_size, 25)

    # Build Policy Processors
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy,
        pretrained_path=hf_policy_repo_id,
        dataset_stats=dataset.meta.stats,
        # The inference device is automatically set to match the detected hardware, overriding any previous device settings from training to ensure compatibility.
        preprocessor_overrides={"device_processor": {"device": str(policy.config.device)}},
    )

    # Connect to the robot
    robot.connect()

    # Initialize the keyboard listener
    listener, events = init_keyboard_listener()

    if not robot.is_connected:
        raise ValueError("Robot is not connected!")
        
    episode_idx = 0
    while episode_idx < NUM_EPISODES and not events["stop_recording"]:
        log_say(f"Running inference, episode {episode_idx + 1} of {NUM_EPISODES}", play_sounds=PLAY_SOUNDS)

        # Main record loop
        record_loop(
            robot=robot,
            events=events,
            fps=FPS,
            policy=policy,
            preprocessor=preprocessor,  # Pass the pre and post policy processors
            postprocessor=postprocessor,
            dataset=dataset,
            control_time_s=EPISODE_TIME_SEC,
            single_task=TASK_DESCRIPTION,
            display_data=True,
            teleop_action_processor=teleop_action_processor,
            robot_action_processor=robot_action_processor,
            robot_observation_processor=robot_observation_processor,
        )

        # Reset the environment if not stopping or re-recording
        if not events["stop_recording"] and ((episode_idx < NUM_EPISODES - 1) or events["rerecord_episode"]):
            log_say("Reset the environment", play_sounds=PLAY_SOUNDS)
            record_loop(
                robot=robot,
                events=events,
                fps=FPS,
                control_time_s=RESET_TIME_SEC,
                single_task=TASK_DESCRIPTION,
                display_data=False,
                teleop_action_processor=teleop_action_processor,
                robot_action_processor=robot_action_processor,
                robot_observation_processor=robot_observation_processor,
            )

        if events["rerecord_episode"]:
            log_say("Re-run episode", play_sounds=PLAY_SOUNDS)
            events["rerecord_episode"] = False
            events["exit_early"] = False
            dataset.clear_episode_buffer()
            continue

        episode_idx += 1

    # Clean up
    log_say("Stop recording", play_sounds=PLAY_SOUNDS)
    robot.disconnect()
    listener.stop()



if __name__ == "__main__":    
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy_id", type=str, required=True, help="HuggingFace policy name to evaluate")
    parser.add_argument("--dataset_id", type=str, required=False, help="HuggingFace dataset name to use for stats")
    parser.add_argument("--policy_type", type=str, default="act", help="Type of policy to evaluate (default: act)")
    args = parser.parse_args()
    main(args.policy_id, args.dataset_id, args.policy_type)