import argparse

from huggingface_hub import create_repo, HfApi
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import create_lerobot_dataset_card


def parse_args():
    parser = argparse.ArgumentParser(description="Push local dataset to Hugging Face Hub")
    parser.add_argument("--local_path", type=str, required=True, help="Local dataset path")
    parser.add_argument("--repo_id", type=str, required=True, help="Hugging Face dataset name")
    return parser.parse_args()

args = parse_args()

local_path = args.local_path
repo_id = args.repo_id

print(f"Checking local dataset path: {local_path}")
dataset = LeRobotDataset(repo_id=repo_id, root=local_path)

print("Creating destination repo on Hugging Face...")

create_repo(
    repo_id=repo_id,
    repo_type="dataset",
    exist_ok=True,
)

print("Uploading dataset...")

dataset.push_to_hub()
card = create_lerobot_dataset_card(tags=["LeRobot"], dataset_info=dataset.meta.info, license="apache-2.0")
card.push_to_hub(repo_id=repo_id, repo_type="dataset")

print("✅ Dataset uploaded to Hugging Face.")
print(f"\t-> New dataset: https://huggingface.co/datasets/{repo_id}")
