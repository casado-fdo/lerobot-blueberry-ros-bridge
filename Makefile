ROS_MASTER_URI ?= http://127.0.0.1:11311
ROS_IP ?= 127.0.0.1
XSOCK ?= /tmp/.X11-unix
XAUTH ?= /tmp/.docker.xauth
XDG_RUNTIME_DIR ?= /run/user/$(shell id -u)

.build:
	docker build -t lerobot-gr00t:latest -f Dockerfile . 

.start_if_not_running:
	@if ! docker ps --format '{{.Names}}' | grep -q '^lerobot$$'; then \
		echo 'Starting lerobot container...'; \
		$(MAKE) start; \
	fi

start:
	@xhost +si:localuser:root >> /dev/null
	docker run --rm --detach \
		--privileged \
		-e DISPLAY \
		-e ROS_MASTER_URI=${ROS_MASTER_URI} \
		-e ROS_IP=${ROS_IP} \
		-e "NVIDIA_VISIBLE_DEVICES=all" \
		-e "NVIDIA_DRIVER_CAPABILITIES=all" \
		--env-file $(ENV_FILE) \
		-e XAUTHORITY=${XAUTH} \
		-e PULSE_SERVER=unix:${XDG_RUNTIME_DIR}/pulse/native \
		-v ${XDG_RUNTIME_DIR}/pulse:${XDG_RUNTIME_DIR}/pulse \
		-v ${XSOCK}:${XSOCK} \
		-v ${XAUTH}:${XAUTH} \
		-v ./data:/data \
		-v ./lerobot_robot_ros:/workspace/lerobot_robot_ros \
		-v ./lerobot_teleoperator_ros:/workspace/lerobot_teleoperator_ros \
		-v ./scripts:/workspace/scripts \
		-v /dev:/dev \
		-v /tmp/argus_socket:/tmp/argus_socket \
		-v /home/${USER}/.cache/at-spi:/home/${USER}/.cache/at-spi \
		--net host \
		--gpus all \
		--runtime nvidia \
		-t \
		--name lerobot lerobot-gr00t:latest

record: .start_if_not_running
	docker exec -it lerobot \
		bash -c "hf auth login --token ${HUGGINGFACE_HUB_TOKEN} && python3 scripts/pl_neon_to_v4l2_streamer.py & python3 scripts/data_collector.py"

debug: .start_if_not_running
	docker exec -it lerobot bash

stop:
	docker stop lerobot