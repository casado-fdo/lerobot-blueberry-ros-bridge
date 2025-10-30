ROS_MASTER_URI ?= http://127.0.1:11311
ROS_IP ?= 127.0.0.1

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
		-v /tmp/.X11-unix:/tmp/.X11-unix \
		-v ./lerobot_robot_ros:/workspace/lerobot_robot_ros \
		-v ./lerobot_teleoperator_ros:/workspace/lerobot_teleoperator_ros \
		-v ./scripts:/workspace/scripts \
		-v ./data:/workspace/data \
		--net host \
		--gpus all \
		-t \
		--name lerobot lerobot:latest

teleop: .start_if_not_running
	docker exec -it lerobot \
		bash -c "lerobot-teleoperate \
		--robot.type=blueberry \
		--teleop.type=leap_motion_ros \
		--display_data=false"

record: .start_if_not_running
	docker exec -it lerobot \
		bash -c "lerobot-record \
			--robot.type=blueberry \
			--dataset.repo_id='.data/record-test' \
			--dataset.single_task='test' \
			--teleop.type=leap_motion_ros \
			--display_data=false"

debug: .start_if_not_running
	docker exec -it lerobot bash

stop:
	docker stop lerobot