# System and utils for preprocessing
import logging
import os
from pathlib import Path

# Deep learning libs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import wandb
from PIL import Image
from rich.progress import (BarColumn, Progress, SpinnerColumn, TextColumn,
                           TimeElapsedColumn, TimeRemainingColumn)
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from tqdm import tqdm

# Custom libs
from DataLoader import RSNATestDataset, RSNATrainDataset
from Model import ResNet, Block


# Pre-initializing the loggers
progress = Progress(
    SpinnerColumn(finished_text="[bold blue]:heavy_check_mark:", style='blue'),
    TextColumn("[bold blue]{task.description}", justify="right"),
    BarColumn(bar_width=50),
    "[progress.percentage]{task.percentage:>3.1f}%",
    "•",
    TimeRemainingColumn(),
    "•",
    TimeElapsedColumn(),
    "•",
    "[progress.filesize]Passed: {task.completed} item",
    "•",
    "[progress.filesize.total]Total: {task.total} item",
)

wandb.login(key='0257777f14fecbf445207a8fdacdee681c72113a')


def train_net(net, device, train_loader, test_loader, 
            epochs, batch_size, learning_rate,
            amp: bool = False, save_checkpoint: bool = True, 
            dir_checkpoint:str = './checkpoints/'):



    # Defining the optimizer
    optimizer = torch.optim.RMSprop(net.parameters(), 
                            lr=learning_rate, 
                            weight_decay=1e-8, 
                            momentum=0.9)
    # Defining the scheduler
    # goal: maximize Dice score
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 
                                                    'max', patience=2)
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
    # Defining the loss function
    criterion = torch.nn.BCEWithLogitsLoss()

    # Defining the global step
    global_step = 0
    n_train = len(train_loader.dataset)

    # Setting up the wandb logger
    experiment = wandb.init(project="Bone-Age-RSNA", entity="rsna-bone-age")
    experiment.config.update(dict(epochs=epochs, batch_size=batch_size, learning_rate=learning_rate,
                            save_checkpoint=save_checkpoint, amp=amp))

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {learning_rate}
        Training size:   {n_train}
        Checkpoints:     {save_checkpoint}
        Device:          {device}
        Mixed Precision: {amp}
    ''')

    # Begin training
    for epoch in range(epochs):
        net.to(device=device, dtype=torch.float32)
        net.train()
        epoch_loss = 0
        
        # with progress:
        #     for _, img, boneage, sex, num_classes in progress.track(train_loader):
        
        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
            for _, images, boneage, sex, num_classes in train_loader:

                images = torch.unsqueeze(images, 1)

                assert images.shape[1] == net.n_channels, \
                    f'Network has been defined with {net.n_channels} input channels, ' \
                    f'but loaded images have {images.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'

                images = images.to(device=device, dtype=torch.float32)

                boneage_onehot = torch.nn.functional.one_hot(torch.tensor(boneage), num_classes = int(num_classes))
                age = boneage_onehot.to(device=device, dtype=torch.float32)


                # Forward pass
                with torch.cuda.amp.autocast(enabled=amp):
                    age_pred = net(images)
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
                experiment.log({
                    'train loss': loss.item(),
                    'step': global_step,
                    'epoch': epoch
                })
                pbar.set_postfix(**{'loss (batch)': loss.item()})

        if save_checkpoint:
            Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
            torch.save(net.state_dict(), str(f"{dir_checkpoint}/checkpoint_epoch{epoch + 1}.pth"))    



def ResNet50(img_channel=3, num_classes=1000):
    return ResNet(Block, [3, 4, 6, 3], img_channel, num_classes)

def ResNet101(img_channel=3, num_classes=1000):
    return ResNet(Block, [3, 4, 23, 3], img_channel, num_classes)


if __name__ == '__main__':
    
    torch.cuda.empty_cache()
    torch.cuda.memory_summary(device=None, abbreviated=False)
    
    defualt_path = ''
    train_dataset = RSNATrainDataset(data_file = Path(defualt_path, 'dataset/rsna-bone-age/boneage-training-dataset.csv'),
                           image_dir = Path(defualt_path, 'dataset/rsna-bone-age/boneage-training-dataset/boneage-training-dataset/'),
                           basedOnSex=True, gender='female')

    test_dataset = RSNATestDataset(data_file = defualt_path + 'dataset/rsna-bone-age/boneage-test-dataset.csv',
                           image_dir = defualt_path + 'dataset/rsna-bone-age/boneage-test-dataset/boneage-test-dataset/',
                           basedOnSex=True, gender='male')
    
    num_classes = train_dataset.num_classes                   
    net = ResNet50(img_channel=1, num_classes=num_classes)
    device = 'cuda'
    learning_rate = 0.0001
    epochs = 2
    batch_size = 1
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False, num_workers=1, pin_memory=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=1, pin_memory=True)

    train_net(net, device, train_loader, test_loader, 
            epochs, batch_size, learning_rate)