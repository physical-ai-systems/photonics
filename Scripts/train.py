import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import logging
import math
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from Utils.logger import setup_logger
from Utils.Utils import save_checkpoint
from Utils.optimizers import configure_optimizers
from Utils.training import train_one_epoch
from Utils.testing import test_one_epoch
from Utils.args import train_options
from Utils.config import model_config
from Models.get_model import get_model, get_schedulers
from Utils.Utils import setup_environment
from Dataset.TMM_Fast import PhotonicDatasetTMMFast



def main():

    args = train_options()
    repo_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_config_path = os.path.join(repo_path, 'configs', 'models', args.config + '.yaml')
    config = model_config(model_config_path, args)

    setup_environment(args.seed)

    os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'DETAIL'
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)

    accelerator = Accelerator(
        mixed_precision=args.mixed_precision, 
        gradient_accumulation_steps=1,
        kwargs_handlers=[ddp_kwargs],
    )
    accelerator.even_batches = False
    device = accelerator.device
    print(f"Using device: {device}")
    if not torch.cuda.is_available():
        device = 'cpu'

    experiment_path = os.path.join(args.main_path, 'experiments', args.experiment)
    
    if accelerator.is_main_process:
        os.makedirs(experiment_path, exist_ok=True)

    accelerator.wait_for_everyone()

    setup_logger('train', experiment_path, 'train_' + args.experiment, level=logging.INFO,
                        screen=True, tofile=True)
    setup_logger('val', experiment_path, 'val_' + args.experiment, level=logging.INFO,
                        screen=True, tofile=True)
    
    logger_train = logging.getLogger('train')
    logger_val = logging.getLogger('val')

    if accelerator.is_main_process:
        logger_train.info(f"Using device: {accelerator.device}")

    tb_logger = SummaryWriter(log_dir=experiment_path) if accelerator.is_main_process else None

    checkpoint_path = os.path.join(experiment_path, 'checkpoints')

    if accelerator.is_main_process:
        os.makedirs(checkpoint_path, exist_ok=True)
    
    train_dataset = PhotonicDatasetTMMFast( **args.dataset, batch_size=args.batch_size, device=device)
    test_dataset = PhotonicDatasetTMMFast( **args.dataset, batch_size=args.test_batch_size, test_mode=True, device=device)
    

    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=0, collate_fn=lambda x: x[0])
    test_dataloader  = DataLoader(test_dataset,  batch_size=1, shuffle=False, num_workers=0, collate_fn=lambda x: x[0])

    net, _,  criterion = get_model(config, args, device)
    optimizer = configure_optimizers(net, args)
    lr_scheduler = get_schedulers(optimizer, args)
    

    net, criterion, optimizer, train_dataloader, test_dataloader, lr_scheduler = accelerator.prepare(
        net, criterion, optimizer, train_dataloader, test_dataloader, lr_scheduler
    )


    best_checkpoint = os.path.join(experiment_path, 'checkpoints', 'checkpoint_best_loss')
    if os.path.exists(best_checkpoint):
        accelerator.load_state(best_checkpoint,strict=False)
        metadata_path = os.path.join(best_checkpoint, 'metadata.pth')
        if os.path.exists(metadata_path):
            metadata = torch.load(metadata_path, map_location=device)
            start_epoch = metadata['epoch']
            best_loss = metadata['loss']
            current_step = start_epoch * math.ceil(len(train_dataloader.dataset) / args.batch_size)
            if accelerator.is_main_process:
                print(f"Loaded checkpoint from epoch {start_epoch} with loss {best_loss}")
        else:
            start_epoch = 0
            best_loss = 1e10
            current_step = 0
    else:
        start_epoch = 0
        best_loss = 1e10
        current_step = 0

    if accelerator.is_main_process:
        logger_train.info(args)
        logger_train.info(config)
        logger_train.info(net)
        logger_train.info(optimizer)

    for epoch in range(start_epoch, args.epochs):
        current_step = train_one_epoch(
            net,
            criterion,
            train_dataloader,
            optimizer,
            epoch,
            args.clip_max_norm,
            logger_train,
            tb_logger,
            current_step,
            accelerator,
        )

        try:
            loss = test_one_epoch(epoch, test_dataloader, net, criterion, logger_val, tb_logger, accelerator)
        except Exception as e:
            logger_val.error(f"Validation failed at epoch {epoch}: {e}")
            loss = float('inf')

        lr_scheduler.step(epoch)
            
        if accelerator.is_main_process:
            logger_train.info(f"Epoch {epoch+1}/{args.epochs} learning rate: {lr_scheduler.get_last_lr()[0]:.6f}")
            # Save checkpoints
            is_best = loss < best_loss
            best_loss = min(loss, best_loss)
            if (epoch + 1) % args.save_every == 0:
                save_checkpoint(
                    accelerator,
                    {
                        "epoch": epoch + 1,
                        "loss": loss,
                    },
                    is_best,
                    checkpoint_path,
                    model=net
                )
                if is_best:
                    logger_val.info('best checkpoint saved.')
    
        accelerator.wait_for_everyone()

if __name__ == '__main__':
    main()
