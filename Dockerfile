FROM dustynv/realsense:r36.2.0

RUN apt-get update && apt upgrade -y && apt-get install -y --upgrade \
    build-essential cmake net-tools iputils-ping \
    python3-dev python3-pip pkg-config libavformat-dev \
    libavcodec-dev libavdevice-dev libavutil-dev \
    libswscale-dev libswresample-dev libavfilter-dev \
    libglib2.0-0 ffmpeg speech-dispatcher libgeos-dev \
    libssl-dev libusb-1.0-0-dev pkg-config udev \
    libudev-dev libglfw3-dev libgl1-mesa-dev libglu1-mesa-dev \
    curl lsb-release git

# Update the index URLs to point to the JetPack 7 / CUDA compatible repos
ENV PIP_INDEX_URL=https://pypi.jetson-ai-lab.io/sbsa/cu128
ENV PIP_EXTRA_INDEX_URL=https://pypi.org/simple

# The installer requires curl (and certificates) to download the release archive
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates
RUN pip install --upgrade pip

# Install extra dependencies for rerun and text-to-speech inside Docker
RUN apt-get update && \
    apt-get install -y python3-pip libgtk-3-dev libxkbcommon-x11-0 vulkan-tools mpg123
RUN pip install gTTS

# Install LeRobot and dependencies
RUN git clone https://github.com/huggingface/lerobot.git
RUN cd lerobot && \
    pip install -e . 

# Install rospypi to interface with ROS Noetic
RUN pip install --extra-index-url https://rospypi.github.io/simple/ rospy-all

# Set up workspace
WORKDIR /workspace

# Create data directory
RUN mkdir -p data

# Copy lerobot custom HW packages
COPY lerobot_robot_ros /workspace/lerobot_robot_ros/
COPY lerobot_teleoperator_ros /workspace/lerobot_teleoperator_ros/

# Install lerobot custom HW packages
RUN pip install -e lerobot_robot_ros 
RUN pip install -e lerobot_teleoperator_ros

# Copy scripts
COPY scripts /workspace/scripts/

CMD ["/bin/bash"]