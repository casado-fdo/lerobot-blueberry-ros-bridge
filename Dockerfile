FROM pytorch/pytorch:2.9.0-cuda12.8-cudnn9-runtime

RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    python3-dev pkg-config libavformat-dev \
    libavcodec-dev libavdevice-dev libavutil-dev \
    libswscale-dev libswresample-dev libavfilter-dev \
    libglib2.0-0 libgl1-mesa-glx libegl1-mesa ffmpeg \
    speech-dispatcher libgeos-dev

# Install LeRobot and dependencies
RUN pip3 install --upgrade pip && \
    pip3 install 'lerobot[all]'

# Install rospypi to interface with ROS Noetic
RUN pip3 install --extra-index-url https://rospypi.github.io/simple/ rospy-all

# Set up workspace
WORKDIR /workspace

# Create data directory
RUN mkdir -p data

# Copy lerobot custom HW packages
COPY lerobot_robot_ros /workspace/lerobot_robot_ros/
COPY lerobot_teleoperator_ros /workspace/lerobot_teleoperator_ros/

# Install lerobot custom HW packages
RUN pip3 install -e lerobot_robot_ros
RUN pip3 install -e lerobot_teleoperator_ros

# Copy scripts
COPY scripts /workspace/scripts/

CMD ["/bin/bash"]