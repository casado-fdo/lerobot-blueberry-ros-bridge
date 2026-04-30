# LeRobot Blueberry ROS Bridge

A ROS bridge for LeRobot that enables integration with custom robotic platforms, specifically designed for the Blueberry smart robotic wheelchair but can be used as a reference to integrate other robotic platforms.


## Overview

The LeRobot ROS Bridge connects LeRobot's machine learning capabilities with ROS, supporting:
- **Blueberry Platform**: Custom dual-arm robotic system with Kinova arms running in ROS Noetic
- **Leap Motion Teleoperation**: Hand tracking and gesture-based control
- **Pupil Labs Neon Streaming**: Real-time FPV video streamming using Gstreamer/V4L2 and eye gaze tracking

Other robots and teleoperation interfaces can be added by extending the `lerobot_robot_ros` and `lerobot_teleoperator_ros` packages.

## Main Features

- **Data Collection**: Automated dataset recording with Hugging Face Hub integration
- **Async Inference**: gRPC-based policy server for low-latency execution
- **Docker Support**: Complete containerized environment for Jetson platforms


## Quick Start

### Prerequisites

- NVIDIA Jetson device (tested on AGX Thor for Blueberry robot)
- Docker and NVIDIA Container Toolkit
- Hugging Face account and access token
- Blueberry robot hardware with Kinova dual-arm system

### Installation

1. **Clone the repository**:
   ```bash
   git clone git@github.com:casado-fdo/lerobot-ros-bridge.git
   cd lerobot-ros-bridge
   ```

2. **Set up environment**:
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

3. **Prepare Jetson environment** (required before building):
   ```bash
   # Copy Jetson release file
   sudo cp /etc/nv_tegra_release .
   
   # Build base LeRobot image
   git clone https://github.com/dusty-nv/jetson-containers
   bash jetson-containers/install.sh
   jetson-containers build lerobot
   ```

4. **Build and run**:
   ```bash
   make .build
   make start
   ```

## Usage

### Data Collection

Record training data with teleoperation:

```bash
make record
```

This will:
- Start the Blueberry robot interface
- Enable Leap Motion teleoperation
- Record episodes at specified FPS
- Upload to Hugging Face Hub (optional)

### Policy Evaluation

Run trained policies asynchronously:

```bash
make eval
```

This starts the policy server on port 8090 with 15 FPS inference.

### Debug and Development

Enter the container for debugging:

```bash
make debug
```

Stop all services:

```bash
make stop
```

## Configuration

### Environment Variables

Key configuration options in `.env`:

```bash
# Hugging Face Integration
HUGGINGFACE_HUB_TOKEN=your_token_here
HUGGINGFACE_USERNAME=your_username
HUGGINGFACE_DATASET_NAME=default

# Recording Settings
RECORDING_FPS=30
RECORDING_NUM_EPISODES=1
RECORDING_EPISODE_TIME_SEC=30
RECORDING_RESET_TIME_SEC=5
RECORDING_VIDEO_WIDTH=640
RECORDING_VIDEO_HEIGHT=480

# Task Configuration
RECORDING_TASK_DESCRIPTION="Task description"
RECORDING_PLAY_SOUNDS=true
RECORDING_ENABLE_RERUN=false
```

### ROS Topics

The bridge interfaces with these ROS topics:

- `/l_kinova_/leap_teleop/cartesian_velocity` - Left arm velocity commands
- `/r_kinova_/leap_teleop/cartesian_velocity` - Right arm velocity commands  
- `/left_hand/leap_teleop/hand_angles` - Left hand joint angles
- `/right_hand/leap_teleop/hand_angles` - Right hand joint angles

## Scripts

### Core Scripts

- **`data_collector.py`** - Main data collection with teleoperation
- **`eval_policy.py`** - Synchronous policy evaluation
- **`eval_policy_async.py`** - Asynchronous policy evaluation
- **`replay_episode.py`** - Replay recorded episodes
- **`resize_dataset.py`** - Resize dataset videos to specified resolution
- **`push_local_dataset_to_hub.py`** - Upload datasets to Hugging Face



## Docker Environment

The Dockerfile includes:

- **Base**: LeRobot Jetson image (tested on AGX Thor Developer Kit - Jetpack 7.0 [L4T 38.2.2] with CUDA 13.0)
- **Dependencies**: ROS Noetic (via `rospypi simple`), RealSense SDK, OpenCV, GStreamer
- **ML Libraries**: PyTorch, Flash Attention, PEFT

### Volume Mounts

- `./data:/data` - Dataset storage
- `./lerobot_robot_ros:/workspace/lerobot_robot_ros` - Robot interface
- `./lerobot_teleoperator_ros:/workspace/lerobot_teleoperator_ros` - Teleoperator
- `./scripts:/workspace/scripts` - Utility scripts
- `/dev:/dev` - Hardware device access

## Development

### Adding New Robots

1. Create robot interface in `lerobot_robot_ros/`
2. Implement `LeRobotROS` base class
3. Add configuration in `__init__.py`
4. Update Dockerfile if new dependencies needed

### Adding New Teleoperators

1. Create teleoperator in `lerobot_teleoperator_ros/`
2. Implement `LeRobotROSTeleop` base class
3. Define ROS topics and message types
4. Add configuration options


## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.