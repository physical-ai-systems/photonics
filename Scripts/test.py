import os
import sys

import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import logging
from torch.utils.data import DataLoader
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from Utils.logger import setup_logger
from Utils.testing import evaluate_test_set
from Utils.args import test_options
from Utils.config import model_config
from Models.get_model import get_model
from Utils.Utils import setup_environment
from Dataset.TMM_Fast import PhotonicDatasetTMMFast

def main():
    args = test_options()
    repo_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_config_path = os.path.join(repo_path, 'configs', 'models', args.config + '.yaml')
    config = model_config(model_config_path, args)

    setup_environment(args.seed)

    # Initialize Accelerator
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        kwargs_handlers=[ddp_kwargs],
    )
    device = accelerator.device
    if not torch.cuda.is_available():
        device = 'cpu'

    
    experiment_path = os.path.join(args.main_path, 'experiments', args.experiment)
    
    # Setup logger
    setup_logger('test', experiment_path, 'test_' + args.experiment, level=logging.INFO, screen=True, tofile=True)
    logger_test = logging.getLogger('test')
    
    if accelerator.is_main_process:
        logger_test.info(f"Using device: {device}")

    # Load Test Dataset
    test_dataset = PhotonicDatasetTMMFast( **args.dataset, batch_size=args.test_batch_size, test_mode=True, device=device)
    test_dataloader  = DataLoader(test_dataset,  batch_size=1, shuffle=False, num_workers=0, collate_fn=lambda x: x[0])

    # Load Model
    net, _,  criterion = get_model(config, args, device)
    
    # Prepare with Accelerator
    net, criterion, test_dataloader = accelerator.prepare(net, criterion, test_dataloader)

    # Load Checkpoint
    checkpoint_path = os.path.join(experiment_path, 'checkpoint_best_loss')
    if os.path.exists(checkpoint_path):
        try:
            accelerator.load_state(checkpoint_path)
            if accelerator.is_main_process:
                logger_test.info(f"Loaded checkpoint from {checkpoint_path}")
        except Exception as e:
            logger_test.error(f"Failed to load checkpoint: {e}")
    else:
        logger_test.warning(f"Checkpoint not found at {checkpoint_path}, using random weights")

    # Run Test
    evaluate_test_set(net, test_dataloader, experiment_path, accelerator, logger_test)

if __name__ == '__main__':
    main()
