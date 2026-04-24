from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import hw_to_dataset_features
from lerobot_robot_ros import BlueberryROS, BlueberryROSConfig
from lerobot_teleoperator_ros import BlueberryTeleop, BlueberryTeleopConfig
from lerobot.utils.control_utils import init_keyboard_listener
from lerobot.utils.visualization_utils import init_rerun
from lerobot.scripts.lerobot_record import record_loop
from lerobot.processor import make_default_processors
from lerobot.utils.constants import ACTION, OBS_STR
from lerobot.configs.types import PolicyFeature, FeatureType
from io_manager import IOManager
import time, os
import rerun as rr
import traceback

NUM_EPISODES = int(os.getenv("RECORDING_NUM_EPISODES", "1"))
FPS = int(os.getenv("RECORDING_FPS", "30"))
EPISODE_TIME_SEC = int(os.getenv("RECORDING_EPISODE_TIME_SEC", "10"))
RESET_TIME_SEC = int(os.getenv("RECORDING_RESET_TIME_SEC", "5"))
TASK_DESCRIPTION = os.getenv("RECORDING_TASK_DESCRIPTION", "No task description provided.")
PLAY_SOUNDS = bool(os.getenv("RECORDING_PLAY_SOUNDS", "True").lower() in ("true", "1", "yes"))
ENABLE_RERUN = bool(os.getenv("RECORDING_ENABLE_RERUN", "False").lower() in ("true", "1", "yes"))
RERUN_IP = os.getenv("RERUN_IP", "127.0.0.1")
RERUN_PORT = int(os.getenv("RERUN_PORT", "9876"))
HF_USERNAME = os.getenv("HUGGINGFACE_USERNAME", "your-username-here")
HF_DATASET_NAME = os.getenv("HUGGINGFACE_DATASET_NAME", "default")
HF_REPO_ID = f"{HF_USERNAME}/{HF_DATASET_NAME}"


def main():
    # I/O setup
    io = IOManager(audio_enabled=PLAY_SOUNDS, tts_engine="gtts")
    io.notify(io.UPDATE, "Initialising data collection...")
    io.log(f"Writing to dataset: {HF_REPO_ID}")

    # Create the robot and teleoperator configurations
    robot_config = BlueberryROSConfig() # default config
    teleop_config = BlueberryTeleopConfig(
        id="blueberry_teleop",
        left_arm_topic="/l_kinova_/leap_teleop/cartesian_velocity",
        right_arm_topic="/r_kinova_/leap_teleop/cartesian_velocity",
        left_hand_topic="/left_hand/leap_teleop/hand_angles",
        right_hand_topic="/right_hand/leap_teleop/hand_angles",
        base_topic="/rnet/pedals/joy",
    )

    # Initialize the robot and teleoperator
    robot = BlueberryROS(robot_config)
    teleop = BlueberryTeleop(teleop_config)

    teleop_action_processor, robot_action_processor, robot_observation_processor = make_default_processors()

    # Configure the dataset features
    action_features = hw_to_dataset_features(robot.action_features, ACTION)
    obs_features = hw_to_dataset_features(robot.observation_features, OBS_STR)
    dataset_features = {**action_features, **obs_features}

    folder_name = time.strftime("%Y%m%d-%H%M%S")

    # Pull existing dataset or create a new one
    try:
        dataset = LeRobotDataset(repo_id=HF_REPO_ID)
    except:
        io.log(f"Dataset {HF_REPO_ID} does not exist. Creating a new one.", speak=False)
        dataset = LeRobotDataset.create(
            repo_id=HF_REPO_ID,
            root=folder_name,
            fps=FPS,
            features=dataset_features,
            robot_type=robot.name,
            use_videos=True,
            image_writer_processes=0,
            image_writer_threads=16
        )

    # Connect the robot and teleoperator
    robot.connect()
    teleop.connect()

    # Initialize the keyboard listener and rerun visualization
    listener, events = init_keyboard_listener()
    if ENABLE_RERUN:
        init_rerun(session_name="blueberry_recording", ip=RERUN_IP, port=RERUN_PORT)

    if not robot.is_connected or not teleop.is_connected:
        raise ValueError("Robot or teleop is not connected!")

    io.notify(io.UPDATE, "Starting recording loop...")
    episode_idx = dataset.num_episodes
    num_episodes = NUM_EPISODES + episode_idx
    try:
        while episode_idx < num_episodes and not events["stop_recording"]:
            # Wait for teleop device to be alive before starting episode
            teleop_wait_logged = False
            while not teleop.ros_interface.is_device_alive():
                if not teleop_wait_logged:
                    io.log("Waiting for teleoperation data to be available...")
                    teleop_wait_logged = True
                time.sleep(0.5)
                if events["stop_recording"]:
                    break
            if events["stop_recording"]:
                break
                
            io.notify(io.UPDATE, f"Recording episode {episode_idx + 1} out of {num_episodes}")

            record_loop(
                robot=robot,
                events=events,
                fps=FPS,
                teleop=teleop,
                dataset=dataset,
                teleop_action_processor=teleop_action_processor,
                robot_action_processor=robot_action_processor,
                robot_observation_processor=robot_observation_processor,
                control_time_s=EPISODE_TIME_SEC,
                single_task=TASK_DESCRIPTION,
                display_data=ENABLE_RERUN,
                #display_compressed_images=True,
            )

            # Reset the environment if not stopping or re-recording
            if not events["stop_recording"] and (episode_idx < num_episodes - 1 or events["rerecord_episode"]):
                io.notify(io.UPDATE, "Reset the environment")
                record_loop(
                    robot=robot,
                    events=events,
                    fps=FPS,
                    teleop=None,
                    dataset=None,
                    teleop_action_processor=teleop_action_processor,
                    robot_action_processor=robot_action_processor,
                    robot_observation_processor=robot_observation_processor,
                    control_time_s=RESET_TIME_SEC,
                    single_task=TASK_DESCRIPTION,
                    display_data=False,
                )

            if events["rerecord_episode"]:
                io.notify(io.UPDATE, "Re-recording episode")
                events["rerecord_episode"] = False
                events["exit_early"] = False
                dataset.clear_episode_buffer()
                continue

            dataset.save_episode(parallel_encoding=False)
            episode_idx += 1
            time.sleep(1.5)
    except Exception as e:
        io.notify(io.FAIL, "An error occurred")
        print(traceback.format_exc())
        
    # Clean up
    io.notify(io.UPDATE, "Stop recording")
    dataset.finalize()
    dataset.push_to_hub()

    robot.disconnect()
    teleop.disconnect()
    listener.stop()

    


if __name__ == "__main__":
    main()

