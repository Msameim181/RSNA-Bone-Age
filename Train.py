import logging
import sys
from datetime import datetime
from pathlib import Path

# Deep learning libs
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import wandb
# Custom libs
from utils.rich_progress_bar import make_bar
from utils.wandb_logger import wandb_setup
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
    optimizer = torch.optim.RMSprop(
                            net.parameters(), 
                            lr=learning_rate, 
                            weight_decay=1e-8, 
                            momentum=0.9)

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
    wandb_logger = wandb_setup(dict(
        epochs = epochs, 
        batch_size = batch_size, 
        learning_rate = learning_rate,
        save_checkpoint = save_checkpoint, 
        amp = amp,
        model = net.name,
        name = run_name,
        device = device,
        optimizer = optimizer.__class__.__name__,
        criterion = criterion.__class__.__name__))


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
                epoch_loss += loss.item()
                wandb_logger.log({
                    'train loss': loss.item(),
                    'step': global_step,
                    'epoch': epoch
                })
                pbar.set_postfix(**{'loss (batch)': loss.item()})


                # Validation
                validation(wandb_logger, net, device, optimizer, scheduler, criterion, 
                    epochs,global_step,  n_train, batch_size,val_loader, images, 
                    boneage, age_pred, gender)

                net.train()

        # Logging
        wandb_logger.log({
            'epoch loss': epoch_loss,
        })

        # Save the model checkpoint
        if save_checkpoint:
            Path(dir_checkpoint).mkdir(parents = True, exist_ok = True)
            torch.save(net.state_dict(), str(f"{dir_checkpoint}/checkpoint_epoch{epoch + 1}.pth"))


# Validation Worker
def validation(
    wandb_logger, 
    net, 
    device:torch.device, 
    optimizer, 
    scheduler, 
    criterion, 
    epochs:int,
    global_step,  
    n_train:int, 
    batch_size:int,
    val_loader:DataLoader, 
    images:torch.Tensor, 
    boneage:torch.Tensor, 
    age_pred:torch.Tensor, 
    gender:torch.Tensor) -> None:
    """Validation Worker
    """

    # Evaluation round
    # Let's See if is it evaluation time or not
    division_step = (n_train // (10 * batch_size))
    if division_step > 0 and global_step % division_step == 0:

        # WandB Storing the model parameters
        histograms = {}
        for tag, value in net.named_parameters():
            tag = tag.replace('/', '.')
            histograms[f'Weights/{tag}'] = wandb.Histogram(value.data.cpu())
            histograms[f'Gradients/{tag}'] = wandb.Histogram(value.grad.data.cpu())


        # Evaluating the model
        val_loss, acc, _ = validate(net, val_loader, device, criterion)
        # 
        scheduler.step(val_loss)

        # WandB Storing the results
        wandb_logger.log({
            'learning rate': optimizer.param_groups[0]['lr'], 
            'validation Loss': val_loss, 
            'validation Correct': acc, 
            'Correct %': acc * 100,
            'Images': wandb.Image(images.cpu()) if batch_size == 1 else [wandb.Image(image.cpu()) for image in images], 
            'Gender': gender if batch_size == 1 else list(gender), 
            'Age': {
                'True': boneage.float().cpu() if batch_size == 1 else [age.float().cpu() for age in boneage], 
                'Pred': age_pred.argmax(dim=1, keepdim=True)[0].float().cpu() if batch_size == 1 else [age for age in age_pred.argmax(dim=1, keepdim=True).float().cpu()],
            }, 
            'step': global_step, 
            'epoch': epochs, 
            **histograms
        })

        logging.info('Validation completed.')
        logging.info('Result Saved.')

