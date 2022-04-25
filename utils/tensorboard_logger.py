import logging
from pathlib import Path

import numpy as np
from torch.utils.tensorboard import SummaryWriter


def tb_setup(config, log_dir:str = './tensorboard'):
    """
    Setup tensorboard logger
    """
    if not log_dir:
        log_dir = './tensorboard'
    model = config['model']
    name = config['name']
    device = config['device']
    epochs = config['epochs']
    batch_size = config['batch_size']
    learning_rate = config['learning_rate']
    save_checkpoint = config['save_checkpoint']
    amp = config['amp']

    # Create a run
    tb_logger = SummaryWriter(log_dir=Path(log_dir, name))

    tb_logger.add_text(tag='name', text_string=str(name), global_step=0)
    tb_logger.add_text(tag='model', text_string=str(model), global_step=0)
    tb_logger.add_text(tag='device', text_string=str(device), global_step=0)
    tb_logger.add_text(tag='epochs', text_string=str(epochs), global_step=0)
    tb_logger.add_text(tag='batch_size', text_string=str(batch_size), global_step=0)
    tb_logger.add_text(tag='learning_rate', text_string=str(learning_rate), global_step=0)
    tb_logger.add_text(tag='amp', text_string=str(amp), global_step=0)
    tb_logger.add_text(tag='save_checkpoint', text_string=str(save_checkpoint), global_step=0)

    return tb_logger


def tb_log_training_step(tb_logger, loss, global_step, epoch, epoch_loss_step):
     # Logging
    tb_logger.add_scalar('Loss/Step Loss', loss.item(), global_step)
    tb_logger.add_scalar('Loss/Train Loss (Step)', epoch_loss_step, global_step)
    tb_logger.add_scalar('Process/Step', global_step, global_step)
    tb_logger.add_scalar('Process/Epoch', epoch, global_step)


def tb_log_training(tb_logger, epoch_loss, val_loss, epoch):
    # Logging
    tb_logger.add_scalar('Loss/Validation Loss (Epoch)', val_loss, epoch)
    tb_logger.add_scalar('Loss/Train Loss', epoch_loss, epoch)
    tb_logger.add_scalar('Loss/Epoch Loss', epoch_loss, epoch)
    logging.info(f'\nEpoch: {epoch} | Train Loss: {epoch_loss:.4f} | Validation Loss: {val_loss:.4f}\n')
    


def tb_log_validation(tb_logger, optimizer, val_loss, acc, 
    images, batch_size, global_step, epoch):
    # TensorBoard Storing the results
    tb_logger.add_scalar('Process/Learning Rate', optimizer.param_groups[0]['lr'], global_step)
    tb_logger.add_scalar('Loss/Validation Loss (Step)', val_loss, global_step)
    tb_logger.add_scalar('Accuracy/Validation Correct (Step)', acc, global_step)
    tb_logger.add_scalar('Accuracy/Correct %', acc * 100, global_step)
    tb_logger.add_scalar('Process/Step', global_step, global_step)
    tb_logger.add_scalar('Process/Epoch', epoch, global_step)
    # img_batch = images.cpu() if batch_size == 1 else [image.cpu() for image in images]
    # tb_logger.add_images('Data/Images', img_batch, global_step)



if __name__ == '__main__':
    # Test
    tb_setup(dict(
            epochs = 1000, 
            batch_size = 200, 
            learning_rate = 0.001,
            save_checkpoint = './checkpoints/', 
            amp = False,
            model = "net.name",
            name = "run_namessss",
            device = "devicessss",
            optimizer = "optimizer.__class__.__name__",
            criterion = "criterion.__class__.__name__"))
