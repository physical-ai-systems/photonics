def get_unwrapped_model(model):
    """Get the unwrapped model from a potentially wrapped model (e.g., DataParallel, DistributedDataParallel)."""
    if hasattr(model, 'module'):
        return model.module
    return model

def train_one_epoch(
    model, criterion, train_dataloader, optimizer, epoch, clip_max_norm, logger_train, tb_logger, current_step, accelerator
):
    model.train()
    
    for i, batch in enumerate(train_dataloader):
       
        spectrum = batch['R'].float()
        
        with accelerator.autocast():
            optimizer.zero_grad()
            
            output = model(spectrum) 
            
            loss_dict = criterion(output, batch)
            loss = loss_dict["loss"]
            
            accelerator.backward(loss)
            if clip_max_norm > 0:
                accelerator.clip_grad_norm_(model.parameters(), clip_max_norm)
            optimizer.step()

        if current_step % 10 == 0 and accelerator.is_main_process:

            if tb_logger is not None:
                tb_logger.add_scalar('[train]: loss', loss.item(), current_step)
                tb_logger.add_scalar('[train]: loss_thickness', loss_dict["loss_thickness"].item(), current_step)
                tb_logger.add_scalar('[train]: loss_material', loss_dict["loss_material"].item(), current_step)
                tb_logger.add_scalar('[train]: lr', optimizer.param_groups[0]['lr'], current_step)

            logger_train.info(
                f"Train epoch {epoch}: ["
                f"{i:5d}/{len(train_dataloader)}"
                f" ({100. * i / len(train_dataloader):.0f}%)] "
                f'Loss: {loss.item():.4f} | '
                f'Thick: {loss_dict["loss_thickness"].item():.4f} | '
                f'Mat: {loss_dict["loss_material"].item():.4f}'
            )


        current_step += 1
        accelerator.wait_for_everyone()
    return current_step