FROM pytorch/pytorch:2.9.0-cuda12.8-cudnn9-runtime

RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    python3-dev pkg-config libavformat-dev \
    libavcodec-dev libavdevice-dev libavutil-dev \
    libswscale-dev libswresample-dev libavfilter-dev

# Install LeRobot and dependencies
RUN pip3 install --upgrade pip && \
    pip3 install lerobot

# Install rospypi to interface with ROS Noetic
RUN pip3 install --extra-index-url https://rospypi.github.io/simple/ rospy-all

# Set up workspace
WORKDIR /workspace

# Create data directory
RUN mkdir -p data

# Copy scripts
COPY scripts /workspace/scripts/

CMD ["/bin/bash"]