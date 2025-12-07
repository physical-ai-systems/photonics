import torch.nn as nn
from Utils.Utils import Config
import yaml
import torch
def model_config(config_path=None, args=None):
    if config_path is not None:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        
        if args is not None and hasattr(args, 'dataset') and isinstance(args.dataset, dict):
            for key, value in args.dataset.items():
                config[key] = value

        # act to the config
        config["act"] = nn.GELU
        return Config(config)
    else:
        raise ValueError("config path is None")
    
def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
