# !/usr/bin/env python

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.processor import make_default_processors
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies.groot.modeling_groot import GrootPolicy
from lerobot.policies.factory import make_pre_post_processors
from lerobot_robot_ros import BlueberryROS, BlueberryROSConfig
from lerobot.utils.control_utils import init_keyboard_listener
from lerobot.scripts.lerobot_record import record_loop
from huggingface_hub import HfApi
import os
import argparse
from io_manager import IOManager


def main(hf_policy_id: str, 
        hf_dataset_id: str = None, 
        play_sounds: bool = True, 
        num_episodes: int = 1, 
        fps: int = 30, 
        episode_time_sec: int = 10, 
        reset_time_sec: int = 5, 
        task_description: str = "No task description provided.",
        hf_username: str = "your-username-here"):
    
    # I/O setup
    io = IOManager(audio_enabled=play_sounds, tts_engine="gtts")
    io.notify(io.UPDATE, "Initialising policy evaluation...")

    # Initialize the robot config
    robot_config = BlueberryROSConfig() # default config
        
    # Initialize the robot
    robot = BlueberryROS(robot_config)

    teleop_action_processor, robot_action_processor, robot_observation_processor = make_default_processors()

    hf_policy_repo_id = f"{hf_username}/{hf_policy_id}"

    # Fetch model metadata
    model_info = HfApi().model_info(hf_policy_repo_id)
    policy_type = model_info.card_data.model_name
    
    # Fetch the dataset metadata
    if hf_dataset_id is not None:
        hf_dataset_repo_id = f"{hf_username}/{hf_dataset_id}"
    else:
        hf_dataset_repo_id = next((tag.split(":")[1] for tag in model_info.tags if tag.startswith("dataset:")), None)
    dataset = LeRobotDataset(hf_dataset_repo_id)

    # Load the model    
    if policy_type.lower() == "act":
        policy = ACTPolicy.from_pretrained(hf_policy_repo_id)
    elif policy_type.lower() == "smolvla":
        policy = SmolVLAPolicy.from_pretrained(hf_policy_repo_id)
    elif policy_type.lower() == "groot":
        policy = GrootPolicy.from_pretrained(hf_policy_repo_id)
    else:
        raise ValueError(f"Unsupported policy type: {policy_type}")

    # Change some parameters in the policy config
    #policy.config.temporal_ensemble_coeff=0.01
    #policy.temporal_ensembler=ACTTemporalEnsembler(policy.config.temporal_ensemble_coeff, policy.config.chunk_size)
    #policy.config.n_action_steps=1 #min(policy.config.chunk_size, 25)

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

    # Summarise the key information, one item per line
    summary_msg = "Evaluation details:\n"
    summary_msg += f"- Policy: {hf_policy_repo_id}\n"
    summary_msg += f"- Policy Type: {policy_type}\n"
    summary_msg += f"- Dataset: {hf_dataset_repo_id}\n"
    summary_msg += f"- Episodes: {num_episodes}\n"
    summary_msg += f"- Episode Time (s): {episode_time_sec}\n"
    summary_msg += f"- Reset Time (s): {reset_time_sec}\n"
    summary_msg += f"- Task: {task_description}\n"
    summary_msg += f"- FPS: {fps}"
    io.log(summary_msg, speak=False)
        
    episode_idx = 0
    while episode_idx < num_episodes and not events["stop_recording"]:
        io.notify(io.UPDATE, f"Running inference, episode {episode_idx + 1} of {num_episodes}")

        # Main record loop
        record_loop(
            robot=robot,
            events=events,
            fps=fps,
            policy=policy,
            preprocessor=preprocessor,  # Pass the pre and post policy processors
            postprocessor=postprocessor,
            dataset=dataset,
            control_time_s=episode_time_sec,
            single_task=task_description,
            display_data=True,
            teleop_action_processor=teleop_action_processor,
            robot_action_processor=robot_action_processor,
            robot_observation_processor=robot_observation_processor,
        )

        # Reset the environment if not stopping or last episode
        if not events["stop_recording"] and (episode_idx < num_episodes - 1):
            io.notify(io.UPDATE, "Reset the environment")
            record_loop(
                robot=robot,
                events=events,
                fps=fps,
                control_time_s=reset_time_sec,
                single_task=task_description,
                display_data=False,
                teleop_action_processor=teleop_action_processor,
                robot_action_processor=robot_action_processor,
                robot_observation_processor=robot_observation_processor,
            )

        if events["rerecord_episode"]:
            io.notify(io.UPDATE, "Re-run episode")
            events["rerecord_episode"] = False
            events["exit_early"] = False
            continue

        episode_idx += 1

    # Clean up
    io.notify(io.UPDATE, "Stop evaluation")
    robot.disconnect()
    listener.stop()



if __name__ == "__main__":    
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy_id", type=str, default=os.getenv("HUGGINGFACE_MODEL_NAME", "your-policy-id-here"), help="HuggingFace policy name to evaluate")
    parser.add_argument("--dataset_id", type=str, default=None, help="HuggingFace dataset name to use for stats")
    parser.add_argument("--play_sounds", type=bool, default=os.getenv("PLAY_SOUNDS", "true").lower() == "true", help="Play sounds during evaluation (default: true)")
    parser.add_argument("--num_episodes", type=int, default=int(os.getenv("RECORDING_NUM_EPISODES", "1")), help="Number of episodes to evaluate (default: 1)")
    parser.add_argument("--fps", type=int, default=int(os.getenv("RECORDING_FPS", "30")), help="Frames per second for evaluation (default: 30)")
    parser.add_argument("--episode_time_sec", type=int, default=int(os.getenv("RECORDING_EPISODE_TIME_SEC", "10")), help="Duration of each episode in seconds (default: 10)")
    parser.add_argument("--reset_time_sec", type=int, default=int(os.getenv("RECORDING_RESET_TIME_SEC", "5")), help="Time to reset between episodes in seconds (default: 5)")
    parser.add_argument("--task_description", type=str, default=os.getenv("RECORDING_TASK_DESCRIPTION", "No task description provided."), help="Task description for evaluation (default: No task description provided.)")
    parser.add_argument("--hf_username", type=str, default=os.getenv("HUGGINGFACE_USERNAME", "your-username-here"), help="HuggingFace username (default: your-username-here)")
    args = parser.parse_args()
    main(args.policy_id, args.dataset_id, args.play_sounds, args.num_episodes, args.fps, args.episode_time_sec, args.reset_time_sec, args.task_description, args.hf_username)