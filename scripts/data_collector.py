#from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import hw_to_dataset_features
from lerobot_robot_ros import ROSRobot, BlueberryROSConfig
from lerobot_teleoperator_ros import LeapMotionROSTeleop, LeapMotionROSTeleopConfig
from lerobot.utils.control_utils import init_keyboard_listener
from lerobot.utils.visualization_utils import init_rerun
from lerobot.scripts.lerobot_record import record_loop
from lerobot.processor import make_default_processors
from lerobot.utils.constants import ACTION, OBS_STR
#from lerobot.utils.utils import log_say
from utils import say, log_say
import time

NUM_EPISODES = 2
FPS = 30
EPISODE_TIME_SEC = 10
RESET_TIME_SEC = 20
TASK_DESCRIPTION = "Test"
HF_REPO_ID = "test/test"
PLAY_SOUNDS = True


# Create the robot and teleoperator configurations
#camera_config = {"front": OpenCVCameraConfig(index_or_path=0, width=640, height=480, fps=FPS)}
robot_config = BlueberryROSConfig(id="blueberry") # , cameras=camera_config)
teleop_config = LeapMotionROSTeleopConfig(id="blueberry_leap_teleop")

# Initialize the robot and teleoperator
robot = ROSRobot(robot_config)
teleop = LeapMotionROSTeleop(teleop_config)

teleop_action_processor, robot_action_processor, robot_observation_processor = make_default_processors()

# Configure the dataset features
action_features = hw_to_dataset_features(robot.action_features, ACTION)
obs_features = hw_to_dataset_features(robot.observation_features, OBS_STR)
dataset_features = {**action_features, **obs_features}

# Debug
#print("Robot observation features:", robot.observation_features)
#print("Robot action features:", robot.action_features)

folder_name = time.strftime("%Y%m%d-%H%M%S")

# Create the dataset
dataset = LeRobotDataset.create(
    repo_id=HF_REPO_ID,
    root=folder_name,
    fps=FPS,
    features=dataset_features,
    robot_type=robot.name,
    use_videos=False,
    image_writer_threads=4,
)

# Connect the robot and teleoperator
robot.connect()
teleop.connect()

# Initialize the keyboard listener and rerun visualization
listener, events = init_keyboard_listener()
init_rerun(session_name="blueberry_recording")

if not robot.is_connected or not teleop.is_connected:
    raise ValueError("Robot or teleop is not connected!")

log_say("Starting recording loop...", play_sounds=PLAY_SOUNDS)
episode_idx = 0
while episode_idx < NUM_EPISODES and not events["stop_recording"]:
    log_say(f"Recording episode {episode_idx + 1} out of {NUM_EPISODES}", play_sounds=PLAY_SOUNDS)

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
        display_data=True,
    )

    # Reset the environment if not stopping or re-recording
    if not events["stop_recording"] and (episode_idx < NUM_EPISODES - 1 or events["rerecord_episode"]):
        log_say("Reset the environment", play_sounds=PLAY_SOUNDS)
        record_loop(
            robot=robot,
            events=events,
            fps=FPS,
            teleop=teleop,
            dataset=dataset,
            teleop_action_processor=teleop_action_processor,
            robot_action_processor=robot_action_processor,
            robot_observation_processor=robot_observation_processor,
            control_time_s=RESET_TIME_SEC,
            single_task=TASK_DESCRIPTION,
            display_data=True,
        )

    if events["rerecord_episode"]:
        log_say("Re-recording episode", play_sounds=PLAY_SOUNDS)
        events["rerecord_episode"] = False
        events["exit_early"] = False
        dataset.clear_episode_buffer()
        continue

    dataset.save_episode()
    episode_idx += 1

# Clean up
log_say("Stop recording", play_sounds=PLAY_SOUNDS)
robot.disconnect()
teleop.disconnect()
listener.stop()

dataset.finalize()
dataset.push_to_hub()



