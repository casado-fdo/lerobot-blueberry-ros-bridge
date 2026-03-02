import shutil
import subprocess
from pathlib import Path
from tqdm import tqdm
import json
import argparse
import pandas as pd
import re

from huggingface_hub import snapshot_download, create_repo, HfApi
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.datasets.utils import create_lerobot_dataset_card


# ============================
# HELPERS
# ============================

def parse_args():
    parser = argparse.ArgumentParser(description="Remove specific features from observation vector")
    parser.add_argument("--src_repo", type=str, required=True, help="Source repository name (e.g., 'username/dataset-name')")
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--features", type=str, nargs='+', help="List of feature names to remove from observation")
    group.add_argument("--regex", type=str, help="Regular expression pattern to match feature names to remove")
    
    return parser.parse_args()

def run(cmd):
    subprocess.run(cmd, check=True)


def main():
    args = parse_args()
    
    src_repo = args.src_repo
    
    dst_repo = src_repo + f"-filtered"

    workdir = Path("./workdir")
    src_root = workdir / "src"
    dst_root = workdir / "dst"

    # ============================
    # 1) DOWNLOAD SOURCE DATASET
    # ============================

    print("Downloading source dataset...")
    src_root.mkdir(parents=True, exist_ok=True)

    snapshot_download(
        repo_id=src_repo,
        repo_type="dataset",
        local_dir=src_root,
        local_dir_use_symlinks=False,
    )

    # Create the destination dataset starting from the source dataset metadata 
    src_metadata = LeRobotDatasetMetadata(src_repo)
    final_dataset = LeRobotDataset.create(repo_id=dst_repo, 
                                root=dst_root,
                                fps=src_metadata.fps,
                                features=src_metadata.features,
                                use_videos=True,)
                          
    # Determine features to remove based on input method
    if args.features:
        features_to_remove = args.features
        print(f"Features to remove (list): {features_to_remove}")
    else:
        regex_pattern = args.regex
        original_names = src_metadata.features["observation.state"]["names"].copy()
        # Escape special characters in regex
        escaped_pattern = re.escape(regex_pattern)
        # Convert wildcard to regex pattern
        regex_pattern = escaped_pattern.replace("\\*", ".*")
        features_to_remove = [name for name in original_names if re.match(regex_pattern, name)]
        print(f"Features to remove (regex): {features_to_remove}")

    print("✅ Downloaded source dataset")

    # ============================
    # 2) COPY NON-VIDEO FILES
    # ============================

    print("Copying data/, meta/, etc.")

    if not dst_root.exists():
        dst_root.mkdir(parents=True, exist_ok=True)

    for item in ["data"]:
        src = src_root / item
        if src.exists():
            if src.is_dir():
                shutil.copytree(src, dst_root / item)
            else:
                shutil.copy2(src, dst_root / item)

    meta_src = src_root / "meta"
    meta_dst = dst_root / "meta"
    for item in ["stats.json", "tasks.parquet", 'info.json']:
        src = meta_src / item
        if src.exists():
            shutil.copy2(src, meta_dst / item)

    episodes_src = src_root / "meta" / "episodes"
    episodes_dst = dst_root / "meta" / "episodes"
    for episode_dir in episodes_src.iterdir():
        if not episode_dir.is_dir():
            continue

        dst_episode_dir = episodes_dst / episode_dir.name
        dst_episode_dir.mkdir(parents=True, exist_ok=True)

        for chunk_file in episode_dir.glob("*.parquet"):
            shutil.copy2(chunk_file, dst_episode_dir / chunk_file.name)

    # ============================
    # 3) UPDATE META/INFO.JSON
    # ============================
    info_json_path = dst_root / "meta" / "info.json"
    if info_json_path.exists():
        with open(info_json_path, "r") as f:
            info = json.load(f)

        # Update observation features
        if "observation.state" in info["features"]:
            original_names = info["features"]["observation.state"]["names"].copy()
            indices_to_keep = [i for i, name in enumerate(original_names) if name not in features_to_remove]
            print("Indices to keep:", indices_to_keep)
            print("Indices to remove:", [i for i, name in enumerate(original_names) if name in features_to_remove])
            
            info["features"]["observation.state"]["names"] = [original_names[i] for i in indices_to_keep]
            info["features"]["observation.state"]["shape"][0] = len(indices_to_keep)
        
        with open(info_json_path, "w") as f:
            json.dump(info, f, indent=4)

    print("✅ meta/info.json updated to match filtered observation space")

    # ============================
    # 4) UPDATE META/STATS.JSON 
    # ============================
    stats_json_path = dst_root / "meta" / "stats.json"
    if stats_json_path.exists():
        with open(stats_json_path, "r") as f:
            stats = json.load(f)
        
        # Update observation statistics by removing specified features
        if "observation.state" in stats:
            observation_stats = stats["observation.state"]            
            # Remove values from each statistical field
            for field in ["min", "max", "mean", "std", "q01", "q10", "q50", "q90", "q99"]:
                if field in observation_stats:
                    observation_stats[field] = [observation_stats[field][i] for i in indices_to_keep]
        
        # Save updated stats
        with open(stats_json_path, "w") as f:
            json.dump(stats, f, indent=4)
        
        print("✅ meta/stats.json updated with filtered observation features")

    # ============================
    # 5) UPDATE EPISODES
    # ============================
    
    print("Updating episodes with filtered observation features...")
    
    data_dir = dst_root / "data"
    if not data_dir.exists():
        print("❌ No observation.state features found in dataset")
        return
    
    # Process all chunk directories
    chunk_dirs = [d for d in data_dir.iterdir() if d.is_dir() and d.name.startswith("chunk-")]
    chunk_dirs.sort()
    
    for chunk_dir in tqdm(chunk_dirs, desc="Processing chunks"):
        # Process all parquet files in the chunk
        parquet_files = list(chunk_dir.glob("*.parquet"))
        parquet_files.sort()
        
        for parquet_file in parquet_files:
            try:
                # Read the parquet file
                df = pd.read_parquet(parquet_file)
                
                # Remove specified observation columns
                columns_to_drop = []
                for feature in features_to_remove:
                    col_name = f"observation.state.{feature}"
                    if col_name in df.columns:
                        columns_to_drop.append(col_name)
                
                if columns_to_drop:
                    df = df.drop(columns=columns_to_drop)
                    
                    # Save the updated parquet file
                    df.to_parquet(parquet_file, index=False)
                
            except Exception as e:
                print(f"❌ Error processing {parquet_file}: {e}")
                continue
    
    print("✅ Episodes updated with filtered observation features")
    

    # ============================
    # 6) PUSH TO HUB
    # ============================

    print("Creating destination repo on Hugging Face...")

    card = create_lerobot_dataset_card(tags=["LeRobot"], dataset_info=info, license="apache-2.0")

    create_repo(
        repo_id=dst_repo,
        repo_type="dataset",
        exist_ok=True,
    )

    print("Uploading filtered dataset...")

    api = HfApi()
    api.upload_folder(
        folder_path=str(dst_root),
        repo_id=dst_repo,
        repo_type="dataset",
    )
    
    final_dataset.push_to_hub()
    card.push_to_hub(repo_id=dst_repo, repo_type="dataset")
    
    print("✅ Dataset filtered and uploaded to Hugging Face.")
    print(f"\t-> New dataset: https://huggingface.co/datasets/{dst_repo}")

    # Remove everything in the working directory
    shutil.rmtree(workdir)
    
if __name__ == "__main__":
    main()