import os

from elliot.run import run_experiment


def main() -> None:
    config = os.getenv("ELLIOT_CONFIG_PATH")
    if config is None:
        return
    run_experiment(config)


if __name__ == "__main__":
    main()
