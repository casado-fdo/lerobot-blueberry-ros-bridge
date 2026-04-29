# !/usr/bin/env python

from lerobot.utils.control_utils import init_keyboard_listener
import os
import argparse
from io_manager import IOManager
from lerobot_robot_ros import BlueberryInference


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
        
    # Initialize inference interface
    try:
        inference = BlueberryInference(hf_username, hf_policy_id, hf_dataset_id, fps)
    except Exception as e:
        io.notify(io.FAIL, f"Failed to initialise inference process: {e}")
        return

    # Initialize the keyboard listener
    listener, events = init_keyboard_listener()

    io.notify(io.UPDATE, "Ready. Starting evaluation...")

    # Summarise the key information, one item per line
    summary_msg = "Evaluation details:\n"
    summary_msg += inference.get_summary() + "\n"
    summary_msg += f"- Episodes: {num_episodes}\n"
    summary_msg += f"- Episode Time (s): {episode_time_sec}\n"
    summary_msg += f"- Reset Time (s): {reset_time_sec}\n"
    summary_msg += f"- Task: {task_description}"
    io.log(summary_msg, speak=False)
        
    episode_idx = 0
    while episode_idx < num_episodes and not events["stop_recording"]:
        io.notify(io.UPDATE, f"Running inference, episode {episode_idx + 1} of {num_episodes}")

        # Main inference loop
        inference.run_inference_loop(events, episode_time_sec, task_description)

        # Reset the environment if not stopping or last episode
        if not events["stop_recording"] and (episode_idx < num_episodes - 1):
            io.notify(io.UPDATE, "Reset the environment")
            inference.reset_environment(events, reset_time_sec)

        if events["rerecord_episode"]:
            io.notify(io.UPDATE, "Re-run episode")
            events["rerecord_episode"] = False
            events["exit_early"] = False
            continue

        episode_idx += 1

    # Clean up
    io.notify(io.UPDATE, "Stopping evaluation...")
    inference.disconnect()
    listener.stop()



if __name__ == "__main__":    
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy_id", type=str, default=os.getenv("HUGGINGFACE_MODEL_NAME", "your-policy-id-here"), help="HuggingFace policy name to evaluate")
    parser.add_argument("--dataset_id", type=str, default=None, help="HuggingFace dataset name to use for stats")
    parser.add_argument("--play_sounds", type=bool, default=os.getenv("PLAY_SOUNDS", "true").lower() == "true", help="Play sounds during evaluation (default: true)")
    parser.add_argument("--num_episodes", type=int, default=int(os.getenv("RECORDING_NUM_EPISODES", "1")), help="Number of episodes to evaluate (default: 1)")
    parser.add_argument("--fps", type=int, default=int(os.getenv("RECORDING_FPS", "15")), help="Frames per second for evaluation (default: 15)")
    parser.add_argument("--episode_time_sec", type=int, default=int(os.getenv("RECORDING_EPISODE_TIME_SEC", "10")), help="Duration of each episode in seconds (default: 10)")
    parser.add_argument("--reset_time_sec", type=int, default=int(os.getenv("RECORDING_RESET_TIME_SEC", "5")), help="Time to reset between episodes in seconds (default: 5)")
    parser.add_argument("--task_description", type=str, default=os.getenv("RECORDING_TASK_DESCRIPTION", "No task description provided."), help="Task description for evaluation (default: No task description provided.)")
    parser.add_argument("--hf_username", type=str, default=os.getenv("HUGGINGFACE_USERNAME", "your-username-here"), help="HuggingFace username (default: your-username-here)")
    args = parser.parse_args()
    main(args.policy_id, args.dataset_id, args.play_sounds, args.num_episodes, args.fps, args.episode_time_sec, args.reset_time_sec, args.task_description, args.hf_username)