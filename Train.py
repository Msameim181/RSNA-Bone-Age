import logging
import sys
from datetime import datetime
from pathlib import Path

# Deep learning libs
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# Custom libs
from utils.rich_progress_bar import make_bar
from utils.wandb_logger import *
from utils.tensorboard_logger import *
from Validation import validate


# Training Worker
def trainer(
    net,
    device:torch.device,
    train_loader:DataLoader, 
    val_loader:DataLoader,
    val_percent:float,
    epochs:int = None, 
    batch_size:int = None, 
    learning_rate:float = None,
    run_name:str = None,
    WandB_usage:bool = False,
    amp:bool = False, 
    save_checkpoint:bool = True, 
    dir_checkpoint:str = './checkpoints/',) -> None:
    """The Trainer for model"""

    # Handling exeptions and inputs
    if not epochs or not batch_size or not learning_rate:
        raise ValueError('Please provide all hyperparameters.')

    if not run_name:
        run_name = datetime.now().strftime("%Y%m%d_%H%M%S") + f"_{net.name}"

    # Defining the optimizer
    optimizer = torch.optim.Adam(
                            net.parameters(), 
                            lr=learning_rate, 
                            weight_decay=1e-8)

    # Defining the scheduler
    # goal: maximize Dice score
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                                                    optimizer, 
                                                    'max', patience=2)
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
    # Defining the loss function
    criterion = torch.nn.BCEWithLogitsLoss()

    # Defining the global step
    global_step = 0
    n_train = len(train_loader.dataset)
    n_val = len(val_loader.dataset)

    # Initiate WandB
    config = dict(
        net = net,
        epochs = epochs, 
        batch_size = batch_size, 
        learning_rate = learning_rate,
        save_checkpoint = save_checkpoint, 
        amp = amp,
        model = net.name,
        name = run_name,
        device = device,
        optimizer = optimizer.__class__.__name__,
        criterion = criterion.__class__.__name__)

    wandb_logger = wandb_setup(config) if WandB_usage else None
    
    tb_logger = tb_setup(config)

    logging.info(f'''Training Settings:
        Device:             {device}
        Model:              {net.name}
        Image Channel:      {net.in_channels}
        Epochs:             {epochs}
        Batch Size:         {batch_size}
        Learning Rate:      {learning_rate}
        Training Size:      {n_train}
        validation Size:    {n_val}
        validation %:       {val_percent}
        Checkpoints:        {save_checkpoint}
        Mixed Precision:    {amp}
        optimizer:          {optimizer.__class__.__name__}
        criterion:          {criterion.__class__.__name__}
    ''')
    logging.info(f'Start training as "{run_name}" ...')

    # Start training

    for epoch in range(epochs):
        net.to(device = device, dtype = torch.float32)
        net.train()
        epoch_loss = 0
        epoch_step = 0
        # Reading data and Training
        with tqdm(total = n_train, desc = f'Epoch {epoch + 1}/{epochs}', unit = 'img') as pbar:
            for _, images, boneage, boneage_onehot, gender, _ in train_loader:

                images = torch.unsqueeze(images, 1)

                assert images.shape[1] == net.in_channels, \
                    f'Network has been defined with {net.in_channels} input channels, ' \
                    f'but loaded images have {images.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'

                images = images.to(device = device, dtype = torch.float32)
                gender = gender.to(device = device, dtype = torch.float32)

                # boneage_onehot = torch.nn.functional.one_hot(torch.tensor(boneage), num_classes = int(num_classes))
                age = boneage_onehot.to(device = device, dtype = torch.float32)


                # Forward pass
                with torch.cuda.amp.autocast(enabled = amp):
                    age_pred = net([images, gender])
                    loss = criterion(age_pred, age)

                # Backward and optimize
                optimizer.zero_grad()
                grad_scaler.scale(loss).backward()
                grad_scaler.step(optimizer)
                grad_scaler.update()


                # Add the loss to epoch_loss
                pbar.update(images.shape[0])
                global_step += 1
                epoch_step += 1
                epoch_loss += loss.item()

                # Logging
                log_epoch_loss = (epoch_loss / (((epoch_step - 1) * batch_size) + (n_train % batch_size))) if n_train // batch_size == (epoch_step - 1) else (epoch_loss / (epoch_step * batch_size))
                # Update the progress bar
                pbar.set_postfix(**{'Step Loss (Batch)': loss.item(), 'Epoch Loss (Train)': log_epoch_loss})
                # Logging
                if WandB_usage:
                    wandb_log_training_step(wandb_logger, loss, global_step, epoch, log_epoch_loss)
                
                tb_log_training_step(tb_logger, loss, global_step, epoch, log_epoch_loss)
                
                # Validation
                val_loss = validation(wandb_logger, tb_logger, net, device, optimizer, scheduler, criterion, 
                    epoch, global_step, epoch_step, n_train, batch_size,val_loader, images, 
                    boneage, age_pred, gender, WandB_usage)
                
                net.train()

        # Logging
        if WandB_usage:
            wandb_log_training(wandb_logger, epoch_loss / n_train, val_loss, epoch)

        tb_log_training(tb_logger, epoch_loss / n_train, val_loss, epoch)

        # Save the model checkpoint
        if save_checkpoint:
            Path(dir_checkpoint).mkdir(parents = True, exist_ok = True)
            torch.save(net.state_dict(), str(f"{dir_checkpoint}/checkpoint_epoch{epoch + 1}.pth"))

    logging.info(f'Finished Training Course. \n')
    tb_logger.close()
    logging.info(f'Shutting Down... \n')

# Validation Worker
def validation(
    wandb_logger, 
    tb_logger,
    net, 
    device:torch.device, 
    optimizer, 
    scheduler, 
    criterion, 
    epoch:int,
    global_step:int, 
    epoch_step:int, 
    n_train:int, 
    batch_size:int,
    val_loader:DataLoader, 
    images:torch.Tensor, 
    boneage:torch.Tensor, 
    age_pred:torch.Tensor, 
    gender:torch.Tensor,
    WandB_usage:bool,
    val_repeat:int = 2) -> None:
    """Validation Worker
    """

    # Evaluation round
    # Let's See if is it evaluation time or not
    n_train_batch = n_train // batch_size
    val_point = [0 if item == val_repeat else ((n_train_batch//val_repeat) * item)  for item in range(1, val_repeat + 1)]
    n_train_batch += 1
    epoch_step = (global_step % n_train_batch) if global_step >= n_train_batch else global_step
    if epoch_step in val_point:

        # WandB Storing the model parameters
        if WandB_usage:
            histograms = wandb_log_histogram(net)


        # Evaluating the model
        val_loss, acc, _ = validate(net, val_loader, device, criterion)
        # 
        scheduler.step(val_loss)

        # WandB Storing the results
        if WandB_usage:
            wandb_log_validation(wandb_logger, optimizer, val_loss, acc, 
                images, batch_size, gender, boneage, age_pred, 
                global_step, epoch, histograms)

        tb_log_validation(tb_logger, optimizer, val_loss, acc, 
            images, batch_size, global_step, epoch, net)

        logging.info('Validation completed.')
        logging.info('Result Saved.')
        return val_loss

    return None
