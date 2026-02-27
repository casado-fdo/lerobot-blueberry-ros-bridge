####################################################################
# Follow these instructions before building the Docker image:
#   1. Copy the nv_tegra_release file from the host system into 
#      the project's folder:
#         cp /etc/nv_tegra_release .
#   2. Build the base image for your specific Jetson specs:
#         git clone https://github.com/dusty-nv/jetson-containers
#         bash jetson-containers/install.sh
#         jetson-containers build lerobot
#   3. Build this Docker image starting from the base image created
#      in the previous step (update the FROM line below if needed)
#####################################################################

FROM lerobot:r38.3.arm64-sbsa-cu130-24.04

RUN apt-get update && apt upgrade -y && apt-get install -y --upgrade \
    build-essential cmake net-tools iputils-ping \
    python3-dev python3-pip pkg-config libavformat-dev \
    libavcodec-dev libavdevice-dev libavutil-dev \
    libswscale-dev libswresample-dev libavfilter-dev \
    libglib2.0-0 ffmpeg speech-dispatcher libgeos-dev \
    libssl-dev libusb-1.0-0-dev pkg-config udev \
    libudev-dev libglfw3-dev libgl1-mesa-dev libglu1-mesa-dev \
    curl lsb-release git ca-certificates
RUN pip install --upgrade pip

# Install extra dependencies for rerun and text-to-speech inside a Docker container
RUN apt-get update && \
    apt-get install -y python3-pip libgtk-3-dev libxkbcommon-x11-0 vulkan-tools mpg123
RUN pip install gTTS

COPY nv_tegra_release /etc/nv_tegra_release

# Install RealSense SDK
WORKDIR /usr/src
RUN git clone https://github.com/IntelRealSense/librealsense.git
WORKDIR /usr/src/librealsense
RUN mkdir -p /tmp/fake-proc/device-tree
RUN echo "NVIDIA Jetson AGX Thor Developer Kit" > /tmp/fake-proc/device-tree/model
RUN sed -i 's|/proc/device-tree|/tmp/fake-proc/device-tree|g' scripts/patch-realsense-ubuntu-L4T.sh
RUN ./scripts/patch-realsense-ubuntu-L4T.sh
RUN mkdir build && cd build && \
    cmake .. \
    -DFORCE_RSUSB_BACKEND=true \
    -DCMAKE_BUILD_TYPE=release \
    -DCHECK_FOR_UPDATES=OFF \
    -DBUILD_PYTHON_BINDINGS=true \
    -DPYTHON_EXECUTABLE=/usr/bin/python3.12 \
    -DBUILD_WITH_CUDA=true 
WORKDIR /usr/src/librealsense/build
RUN make -j$(($(nproc)-1))
RUN make install
RUN cp ../config/99-realsense-libusb.rules /etc/udev/rules.d/

# Install some extra packages for specific policies like SmolVLA
RUN pip install num2words

# Install rospypi to interface with ROS Noetic
RUN pip install --extra-index-url https://rospypi.github.io/simple/ rospy-all

# Set up workspace
WORKDIR /workspace

# GR00T dependencies
RUN pip install peft
RUN git clone https://github.com/Dao-AILab/flash-attention.git
RUN cd flash-attention && python setup.py install
RUN python -c "import flash_attn; print(f'Flash Attention {flash_attn.__version__} imported successfully')"
RUN pip install dm-tree

# Upgrade lerobot sources to the latest available version
RUN git clone https://github.com/huggingface/lerobot.git
RUN cp -r ./lerobot/* /opt/lerobot/

# Install Pupil Labs eye-tracking glasses dependencies
RUN pip install pupil-labs-realtime-api==1.8.0
RUN apt-get update && apt-get install -y \
    libgstreamer1.0-dev \
    libgstreamer-plugins-base1.0-dev \
    gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad \
    gstreamer1.0-plugins-ugly \
    gstreamer1.0-libav \
    gstreamer1.0-tools
RUN apt install -y v4l2loopback-dkms v4l-utils

# Install extra dependencies for asynchronous inference
RUN pip install grpcio

# Copy lerobot custom HW packages
COPY lerobot_robot_ros /workspace/lerobot_robot_ros/
COPY lerobot_teleoperator_ros /workspace/lerobot_teleoperator_ros/

# Install lerobot custom HW packages
RUN pip install -e lerobot_robot_ros 
RUN pip install -e lerobot_teleoperator_ros

# Copy scripts
COPY scripts /workspace/scripts/

CMD ["/bin/bash"]