import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import logging
import math
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torchvision import transforms
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from Utils.logger import setup_logger
from Utils.Utils import save_checkpoint, patch_compressai_buffer_loading
from Utils.optimizers import configure_optimizers
from Utils.training import train_one_epoch, train_one_epoch_direct
from Utils.testing import test_one_epoch, test_one_epoch_direct
from Utils.args import train_options
from Utils.config import model_config
from Models.get_model import get_model, get_schedulers
from Utils.Utils import setup_environment
from Utils.metrics import ImageMetric
from Dataset.Dataset import PhotonicDataset
patch_compressai_buffer_loading()



def main():
    args = train_options()
    repo_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_config_path = os.path.join(repo_path, 'configs', 'models', args.config + '.yaml')
    config = model_config(model_config_path)


    setup_environment(args.seed)

    # Configure DDP to handle unused parameters
    os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'DETAIL'
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)

    # Initialize Accelerator for distributed training
    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,  # 'no', 'fp16', 'bf16'
        gradient_accumulation_steps=1,
        kwargs_handlers=[ddp_kwargs],
    )
    device = accelerator.device


    experiment_path = os.path.join(args.main_path, 'experiments', args.experiment)
    
    # Only create directories on main process
    if accelerator.is_main_process:
        os.makedirs(experiment_path, exist_ok=True)

    # Wait for main process to create directories
    accelerator.wait_for_everyone()

    setup_logger('train', experiment_path, 'train_' + args.experiment, level=logging.INFO,
                        screen=True, tofile=True)
    setup_logger('val', experiment_path, 'val_' + args.experiment, level=logging.INFO,
                        screen=True, tofile=True)

    logger_train = logging.getLogger('train')
    logger_val = logging.getLogger('val')

    # TensorBoard logger only on main process
    tb_logger = SummaryWriter(log_dir=experiment_path) if accelerator.is_main_process else None

    checkpoint_path = os.path.join(experiment_path, 'checkpoints')
    if accelerator.is_main_process:
        os.makedirs(checkpoint_path, exist_ok=True)
    
    if config.Model == 'DirectEncoder':
        # Use PhotonicDataset for DirectEncoder
        train_dataset = PhotonicDataset(
            num_layers=config.dataset_params['num_layers'],
            ranges=tuple(config.dataset_params['ranges']),
            steps=config.dataset_params['steps'],
            dataset_size=config.dataset_params['dataset_size'],
            batch_size=args.batch_size
        )
        # For testing, we can use the same dataset class but maybe different size or seed
        test_dataset = PhotonicDataset(
            num_layers=config.dataset_params['num_layers'],
            ranges=tuple(config.dataset_params['ranges']),
            steps=config.dataset_params['steps'],
            dataset_size=config.dataset_params['dataset_size'] // 10, # Smaller test set
            batch_size=args.test_batch_size
        )
    else:
        # Legacy/Image support
        train_transforms = transforms.Compose(
            [transforms.RandomCrop(args.patch_size), transforms.ToTensor()]
        )
        test_transforms = transforms.Compose(
            [transforms.ToTensor(), transforms.CenterCrop(args.patch_size)]
        )

        csv_file_path = os.path.join(repo_path, 'datasets', 'csv', args.dataset_csv + '.csv')
        dataset_path = args.dataset
        # Assuming ImageFolderCSV is defined somewhere or imported
        # train_dataset = ImageFolderCSV(dataset_path,csv_file=csv_file_path, transform=train_transforms)
        # test_dataset = ImageFolder(os.path.join(dataset_path, 'test'), transform=test_transforms) 
        raise NotImplementedError("ImageFolderCSV not found in context. Only DirectEncoder is supported currently.")
    

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=None if config.Model == 'DirectEncoder' else args.batch_size, # PhotonicDataset handles batching if batch_size passed to init? No, wait.
        # PhotonicDataset __getitem__ returns a BATCH if batch_size is passed to init?
        # Let's check Dataset.py again.
        num_workers=args.num_workers,
        shuffle=True if config.Model != 'DirectEncoder' else False, # PhotonicDataset is random anyway
        pin_memory=True,
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=None if config.Model == 'DirectEncoder' else args.test_batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=True,
    )
    net, vae, criterion = get_model(config, args, device)
    optimizer, aux_optimizer = configure_optimizers(net, args)
    lr_scheduler, lr_scheduler_aux = get_schedulers(optimizer, aux_optimizer, args)
    
    if config.Model == 'DirectEncoder':
        image_metric = None
    else:
        image_metric = ImageMetric()

    # Prepare everything with accelerator
    # Note: PhotonicDataset returns a dict, accelerator.prepare might need care? 
    # Usually it handles dataloaders fine.
    if image_metric is not None:
        net, criterion, optimizer, aux_optimizer, train_dataloader, test_dataloader, lr_scheduler, lr_scheduler_aux, image_metric = accelerator.prepare(
            net, criterion, optimizer, aux_optimizer, train_dataloader, test_dataloader, lr_scheduler, lr_scheduler_aux, image_metric
        )
    else:
        net, criterion, optimizer, aux_optimizer, train_dataloader, test_dataloader, lr_scheduler, lr_scheduler_aux = accelerator.prepare(
            net, criterion, optimizer, aux_optimizer, train_dataloader, test_dataloader, lr_scheduler, lr_scheduler_aux
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

        if config.Model == 'DirectEncoder':
            current_step = train_one_epoch_direct(
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
            loss = test_one_epoch_direct(epoch, test_dataloader, net, criterion, logger_val, tb_logger, accelerator)
        else:
            current_step = train_one_epoch(
                net,
                vae,
                criterion,
                train_dataloader,
                optimizer,
                aux_optimizer,
                epoch,
                args.clip_max_norm,
                logger_train,
                tb_logger,
                current_step,
                accelerator,
            )

            loss = test_one_epoch(epoch, test_dataloader, net, vae, criterion, logger_val, tb_logger, image_metric, accelerator)

        # lr_scheduler.scheduler.last_epoch = epoch
        lr_scheduler.step(epoch)
        # lr_scheduler_aux.scheduler.last_epoch = epoch
        if lr_scheduler_aux is not None:
            lr_scheduler_aux.step(epoch)
            
        if accelerator.is_main_process:
            aux_lr = lr_scheduler_aux.get_last_lr()[0] if lr_scheduler_aux else 0.0
            logger_train.info(f"Epoch {epoch+1}/{args.epochs} learning rate: {lr_scheduler.get_last_lr()[0]:.6f}, aux learning rate: {aux_lr:.6f}")
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
