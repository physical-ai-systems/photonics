import argparse
import yaml
import os


def get_args(config_stage = "train"):
    
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(parent_dir, "configs/general/general.yaml"), "r") as f:
        config = yaml.safe_load(f)
        general_config = config.get("general", {})
        config = config.get(config_stage, {})
        config.update(general_config)
    return config

        



def train_options(args=None):
    parser = argparse.ArgumentParser(description="Training script.")

    # Load all arguments from the config file
    config = get_args("train")


    for key, value in config.items():
        arg_type = type(value)
        if arg_type == list:
            arg_type = type(value[0])
        
        parser.add_argument(
            f"--{key}",
            default=value,
            type=arg_type,
            required=False,
            help=f"{key} (default: %(default)s)"
        )

    args = parser.parse_args(args)
    return args


def test_options(args=None):
    parser = argparse.ArgumentParser(description="Testing script.")

    # Load all arguments from the config file
    config = get_args("test")

    for key, value in config.items():
        arg_type = type(value)
        if arg_type == list:
            arg_type = type(value[0])

        parser.add_argument(
            f"--{key}",
            default=value,
            type=arg_type,
            required=False,
            help=f"{key} (default: %(default)s)"
        )

    args = parser.parse_args(args)
    return args
