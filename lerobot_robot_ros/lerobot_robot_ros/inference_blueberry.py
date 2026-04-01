# !/usr/bin/env python

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.processor import make_default_processors
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies.groot.modeling_groot import GrootPolicy
from lerobot.policies.factory import make_pre_post_processors
from lerobot.scripts.lerobot_record import record_loop
from .robot_blueberry import BlueberryROS
from .config_blueberry import BlueberryROSConfig
from huggingface_hub import HfApi

class BlueberryInference:
    """
    Custom inference interface for running lerobot policies with Blueberry robot.
    """

    def __init__(self, hf_username: str,  hf_policy_id: str, hf_dataset_id: str = None, fps: int = 30):
        # Define main properties
        self.hf_policy_id = hf_policy_id
        self.hf_username = hf_username
        self.hf_policy_repo_id = f"{hf_username}/{hf_policy_id}"
        self.fps = fps

        # Initialize the robot and connect to it
        self.robot = BlueberryROS(BlueberryROSConfig()) # default config
        self.robot.connect()

        if not self.robot.is_connected: 
            raise ValueError("Robot is not connected!")

        # Fetch model metadata
        self.model_info = HfApi().model_info(self.hf_policy_repo_id)
        self.policy_type = self.model_info.card_data.model_name

        # Fetch dataset
        if hf_dataset_id is not None:
            self.hf_dataset_repo_id = f"{hf_username}/{hf_dataset_id}"
        else:
            self.hf_dataset_repo_id = next((tag.split(":")[1] for tag in self.model_info.tags if tag.startswith("dataset:")), None)
        self.dataset = LeRobotDataset(self.hf_dataset_repo_id)

        # Load model and processors
        self.teleop_action_processor, self.robot_action_processor, self.robot_observation_processor = make_default_processors()
        self.policy = self.load_model(self.policy_type, self.hf_policy_repo_id)
        self.preprocessor, self.postprocessor = self.build_policy_processors(self.policy, self.hf_policy_repo_id, self.dataset)        
        

    def get_summary(self):
        summary_msg = f"- Policy: {self.hf_policy_repo_id}\n"
        summary_msg += f"- Policy Type: {self.policy_type}\n"
        summary_msg += f"- Dataset: {self.hf_dataset_repo_id}\n"
        summary_msg += f"- FPS: {self.fps}"
        return summary_msg

    def load_model(self, hf_policy_type: str, hf_policy_id: str):
        if hf_policy_type.lower() == "act":
            policy = ACTPolicy.from_pretrained(hf_policy_id)
        elif hf_policy_type.lower() == "smolvla":
            policy = SmolVLAPolicy.from_pretrained(hf_policy_id)
        elif hf_policy_type.lower() == "groot":
            policy = GrootPolicy.from_pretrained(hf_policy_id)
        else:
            raise ValueError(f"Unsupported policy type: {hf_policy_id}")

        # Change some parameters in the policy config
        #policy.config.temporal_ensemble_coeff=0.01
        #policy.temporal_ensembler=ACTTemporalEnsembler(policy.config.temporal_ensemble_coeff, policy.config.chunk_size)
        #policy.config.n_action_steps=1 #min(policy.config.chunk_size, 25)

        return policy

    def build_policy_processors(self, policy, hf_policy_id, dataset):
        # Build Policy Processors
        preprocessor, postprocessor = make_pre_post_processors(
            policy_cfg=policy,
            pretrained_path=hf_policy_id,
            dataset_stats=dataset.meta.stats,
            # The inference device is automatically set to match the detected hardware, overriding any previous device settings from training to ensure compatibility.
            preprocessor_overrides={"device_processor": {"device": str(policy.config.device)}},
        )
        return preprocessor, postprocessor

    def run_inference_loop(self, events, episode_time_sec, task_description):
        record_loop(
            robot=self.robot,
            events=events,
            fps=self.fps,
            policy=self.policy,
            preprocessor=self.preprocessor,
            postprocessor=self.postprocessor,
            dataset=self.dataset,
            control_time_s=episode_time_sec,
            single_task=task_description,
            display_data=True,
            teleop_action_processor=self.teleop_action_processor,
            robot_action_processor=self.robot_action_processor,
            robot_observation_processor=self.robot_observation_processor,
        )

    def reset_environment(self, events, reset_time_sec):
        record_loop(
            robot=self.robot,
            events=events,
            fps=self.fps,
            control_time_s=reset_time_sec,
            single_task=None,
            display_data=False,
            teleop_action_processor=self.teleop_action_processor,
            robot_action_processor=self.robot_action_processor,
            robot_observation_processor=self.robot_observation_processor,
        )

    def disconnect(self):
        self.robot.disconnect()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect()
