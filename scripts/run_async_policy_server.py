from lerobot_robot_ros.async_policy_server import PolicyServerConfig
from lerobot_robot_ros.async_policy_server import serve


def main():
    host = "127.0.0.1"
    port = 8090

    config = PolicyServerConfig(
        host=host,
        port=port,
    )
    serve(config)


if __name__ == "__main__":
    main()