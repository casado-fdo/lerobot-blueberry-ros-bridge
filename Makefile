ROS_MASTER_URI ?= http://127.0.1:11311
ROS_IP ?= 127.0.0.1
XSOCK ?= /tmp/.X11-unix
XAUTH ?= /tmp/.docker.xauth

.build:
	docker build -t lerobot:latest -f Dockerfile . 

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
		-e HUGGINGFACE_HUB_TOKEN=${HUGGINGFACE_HUB_TOKEN} \
		-e "NVIDIA_VISIBLE_DEVICES=all" \
		-e "NVIDIA_DRIVER_CAPABILITIES=all" \
		-e XAUTHORITY=${XAUTH} \
		-v ${XSOCK}:${XSOCK} \
		-v ${XAUTH}:${XAUTH} \
		-v ./lerobot_robot_ros:/workspace/lerobot_robot_ros \
		-v ./lerobot_teleoperator_ros:/workspace/lerobot_teleoperator_ros \
		-v ./scripts:/workspace/scripts \
		-v ./data:/workspace/data \
		-v /dev:/dev \
		--net host \
		--gpus all \
		-t \
		--name lerobot lerobot:latest

record: .start_if_not_running
	docker exec -it lerobot \
		bash -c "hf auth login --token ${HUGGINGFACE_HUB_TOKEN} && python scripts/data_collector.py"

debug: .start_if_not_running
	docker exec -it lerobot bash

stop:
	docker stop lerobot