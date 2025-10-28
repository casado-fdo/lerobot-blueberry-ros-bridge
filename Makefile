ros_master_uri ?= http://127.0.1:11311
ros_ip ?= 127.0.0.1

.build:
	docker build -t lerobot:latest -f Dockerfile .

.start_if_not_running:
	@if ! docker ps -a | grep -w lerobot; then $(MAKE) start; fi

start:
	@xhost +si:localuser:root >> /dev/null
	docker run -it --rm --privileged \
		-e DISPLAY \
		-e ROS_MASTER_URI=${ros_master_uri} \
		-e ROS_IP=${ros_ip} \
		-v /tmp/.X11-unix:/tmp/.X11-unix \
		-v ./scripts:/workspace/scripts \
		-v ./data:/workspace/data \
		--net host \
		--name lerobot lerobot:latest

debug: .start_if_not_running
	docker exec -it lerobot bash

stop:
	docker stop lerobot