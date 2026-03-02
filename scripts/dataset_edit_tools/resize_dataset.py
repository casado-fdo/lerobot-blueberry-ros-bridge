import shutil
import subprocess
from pathlib import Path
from tqdm import tqdm
import json
import argparse

from huggingface_hub import snapshot_download, create_repo, HfApi
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.datasets.utils import create_lerobot_dataset_card
from lerobot.datasets.dataset_tools import merge_datasets


# ============================
# HELPERS
# ============================

def parse_args():
    parser = argparse.ArgumentParser(description="Resize dataset videos")
    parser.add_argument("--src_repo", type=str, required=True, help="Source repository name (e.g., 'username/dataset-name')")
    parser.add_argument("--target_w", type=int, required=True, help="Target width")
    parser.add_argument("--target_h", type=int, required=True, help="Target height")
    return parser.parse_args()

def run(cmd):
    subprocess.run(cmd, check=True)

def resize_video_ffmpeg(src, dst, target_w, target_h):
    dst.parent.mkdir(parents=True, exist_ok=True)

    
    cmd = [
        "ffmpeg",
        "-y",
        "-i", str(src),
        "-vf", f"scale={target_w}:{target_h}",
        "-c:v", "libsvtav1",
        "-pix_fmt", "yuv420p",
        "-crf", "30",
        "-g", "2",
        "-an",
        str(dst),
    ]
    run(cmd)

def main():
    args = parse_args()
    
    src_repo = args.src_repo
    target_w = args.target_w
    target_h = args.target_h
    dst_repo = src_repo + f"-multi-{target_w}x{target_h}"

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
    dataset = LeRobotDataset.create(repo_id=dst_repo, 
                                root=dst_root,
                                fps=src_metadata.fps,
                                features=src_metadata.features,
                                use_videos=True,)
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

        for key, feat in info["features"].items():
            print("Updating feature:", key)
            if feat["dtype"] == "video":
                feat["shape"] = [target_h, target_w, 3]
                feat["info"]["video.height"] = target_h
                feat["info"]["video.width"] = target_w

        with open(info_json_path, "w") as f:
            json.dump(info, f, indent=4)

    print("✅ meta/info.json updated to match resized videos")

    
    # ============================
    # 4) RESIZE VIDEOS
    # ============================

    print("Resizing videos...")

    videos_src = src_root / "videos"
    videos_dst = dst_root / "videos"

    for cam_dir in videos_src.iterdir():
        if not cam_dir.is_dir():
            continue

        # e.g. observation.images.front
        for chunk_dir in cam_dir.iterdir():
            if not chunk_dir.is_dir():
                continue

            mp4s = list(chunk_dir.glob("*.mp4"))
            if not mp4s:
                continue

            for video in tqdm(mp4s, desc=f"{cam_dir.name}/{chunk_dir.name}"):
                dst_video = (
                    videos_dst
                    / cam_dir.name
                    / chunk_dir.name
                    / video.name
                )
                resize_video_ffmpeg(video, dst_video, target_w, target_h)
    print("✅ Videos resized")

    # ============================
    # 5) RE-INDEX DATASET (OPTIMISE CHUNKING)
    # ============================

    print("Reindexing dataset...")

    final_root = workdir / "final"
    tmp_dataset = LeRobotDataset(repo_id=dst_repo, root=dst_root)
    final_dataset = merge_datasets(datasets=[tmp_dataset], output_repo_id=dst_repo, output_dir=final_root)
    card = create_lerobot_dataset_card(tags=["LeRobot"], dataset_info=final_dataset.meta.info, license="apache-2.0")
   
    print("✅ Dataset reindexed")

    # ============================
    # 6) PUSH TO HUB
    # ============================

    print("Creating destination repo on Hugging Face...")

    create_repo(
        repo_id=dst_repo,
        repo_type="dataset",
        exist_ok=True,
    )

    print("Uploading resized dataset...")

    api = HfApi()
    api.upload_folder(
        folder_path=str(final_root),
        repo_id=dst_repo,
        repo_type="dataset",
    )
    
    final_dataset.push_to_hub()
    card.push_to_hub(repo_id=dst_repo, repo_type="dataset")

    print("✅ Dataset resized and uploaded to Hugging Face.")
    print(f"\t-> New dataset: https://huggingface.co/datasets/{dst_repo}")

    # Remove everything in the working directory
    shutil.rmtree(workdir)
    
if __name__ == "__main__":
    main()
