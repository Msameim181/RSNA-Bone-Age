# System and utils for preprocessing
import logging
import sys
from datetime import datetime
from pathlib import Path

# Deep learning libs
import torch
from rich.progress import (BarColumn, Progress, SpinnerColumn, TextColumn,
                           TimeElapsedColumn, TimeRemainingColumn)
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

import wandb
from MobileNet import MobileNetV2
from ResNet import ResNet18, ResNet34, ResNet50, ResNet101, ResNet152
# Custom libs
from utils.dataloader import RSNATestDataset, RSNATrainDataset
from Validation import validate

# Pre-initializing the loggers
time_now = datetime.now()

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

def validation(net, val_loader, criterion, device, batch_size, 
        n_train, scheduler, experiment, optimizer, global_step, epoch,
        images, boneage, age_pred, gender):

    # Evaluation round
    division_step = (n_train // (10 * batch_size))
    if division_step > 0 and global_step % division_step == 0:
        histograms = {}
        for tag, value in net.named_parameters():
            tag = tag.replace('/', '.')
            histograms[f'Weights/{tag}'] = wandb.Histogram(value.data.cpu())
            histograms[f'Gradients/{tag}'] = wandb.Histogram(value.grad.data.cpu())
        val_score, correct = validate(net, val_loader, device, criterion)
        scheduler.step(val_score)
        logging.info(f'Validation Dice score: {val_score}, Correct: {correct}')
        experiment.log({
            'learning rate': optimizer.param_groups[0]['lr'],
            'validation Loss': val_score,
            'validation Correct': correct,
            'images': wandb.Image(images[0].cpu()),
            'Gender': gender,
            'Age': {
                'True': boneage.float().cpu(),
                'Pred': age_pred.argmax(dim=1, keepdim=True)[0].float().cpu() if batch_size == 1 else age_pred.argmax(dim=1, keepdim=True).float().cpu(),
            },
            'step': global_step,
            'epoch': epoch,
            **histograms
        })

def train_net(net, device, train_loader, val_loader, 
            epochs, batch_size, learning_rate, val_percent,
            amp: bool = False, save_checkpoint: bool = True, 
            dir_checkpoint:str = './checkpoints/', run_name: str = None,):



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
    n_val = len(val_loader.dataset)

    # Setting up the wandb logger
    experiment = wandb.init(
        project = "Bone-Age-RSNA", 
        entity = "rsna-bone-age", 
        name = run_name, 
        tags = ['bone-age', 'rsna', f'{net.name}', f'{run_name}', f'{device}'],)
    experiment.config.update(dict(
        epochs = epochs, 
        batch_size = batch_size, 
        learning_rate = learning_rate,
        save_checkpoint = save_checkpoint, 
        amp = amp,
        model = net.name,
        name = run_name,
        device = device))

    logging.info(f'''Starting training:
        Model:              {net.name}
        Image Channel:      {net.in_channels}
        Epochs:             {epochs}
        Batch Size:         {batch_size}
        Learning Rate:      {learning_rate}
        Training Size:      {n_train}
        validation Size:    {n_val}
        validation %:       {val_percent}
        Checkpoints:        {save_checkpoint}
        Device:             {device}
        Mixed Precision:    {amp}
    ''')

    # Begin training
    for epoch in range(epochs):
        net.to(device = device, dtype = torch.float32)
        net.train()
        epoch_loss = 0

        # with progress:
        #     for _, img, boneage, sex, num_classes in progress.track(train_loader):

        with tqdm(total = n_train, desc = f'Epoch {epoch + 1}/{epochs}', unit = 'img') as pbar:
            for _, images, boneage, boneage_onehot, sex, num_classes in train_loader:

                images = torch.unsqueeze(images, 1)

                assert images.shape[1] == net.in_channels, \
                    f'Network has been defined with {net.in_channels} input channels, ' \
                    f'but loaded images have {images.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'

                images = images.to(device = device, dtype = torch.float32)
                sex = sex.to(device = device, dtype = torch.float32)

                # boneage_onehot = torch.nn.functional.one_hot(torch.tensor(boneage), num_classes = int(num_classes))
                age = boneage_onehot.to(device = device, dtype = torch.float32)


                # Forward pass
                with torch.cuda.amp.autocast(enabled = amp):
                    age_pred = net([images, sex])
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


                validation(net, val_loader, criterion, device, batch_size, 
                        n_train, scheduler, experiment, optimizer, global_step, epoch,
                        images, boneage, age_pred, sex)

                net.train()

        if save_checkpoint:
            Path(dir_checkpoint).mkdir(parents = True, exist_ok = True)
            torch.save(net.state_dict(), str(f"{dir_checkpoint}/checkpoint_epoch{epoch + 1}.pth"))    



def data_organizer(train_dataset, test_dataset, batch_size: int, val_percent: float = 0.2, shuffle: bool = True, num_workers: int = 1):
    """ Generate the train, validation and test dataloader for model.

    Args:
        train_dataset (dataset class): training dataset.
        test_dataset (dataset class): test dataset.
        batch_size (int): batch size.
        val_percent (float, optional): valifation percentage. Defaults to 0.2.
        shuffle (bool, optional): shuffle data. Defaults to True.
        num_workers (int, optional): number of worker. Defaults to 1.

    Returns:
        data loaders: train_loader, val_loader, test_loader
    """
    n_val = int(len(train_dataset) * val_percent)
    n_train = len(train_dataset) - n_val
    train_set, val_set = random_split(train_dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))
    train_loader = DataLoader(dataset = train_set, batch_size = batch_size, shuffle = shuffle, num_workers = num_workers, pin_memory = True)
    val_loader = DataLoader(dataset = val_set, batch_size = 1, shuffle = shuffle, num_workers = num_workers, pin_memory = True)

    test_loader = DataLoader(dataset = test_dataset, batch_size = batch_size, shuffle = shuffle, num_workers = num_workers, pin_memory = True)

    return train_loader, val_loader, test_loader



if __name__ == '__main__':

    logging.basicConfig(level = logging.INFO, format = '%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    torch.cuda.empty_cache()
    torch.cuda.memory_summary(device = None, abbreviated = False)
    
    basedOnSex = False
    gender = 'male'
    defualt_path = ''
    train_dataset = RSNATrainDataset(data_file = Path(defualt_path, 'dataset/rsna-bone-age/boneage-training-dataset.csv'),
                           image_dir = Path(defualt_path, 'dataset/rsna-bone-age/boneage-training-dataset/boneage-training-dataset/'),
                           basedOnSex = basedOnSex, gender = gender)

    test_dataset = RSNATestDataset(data_file = defualt_path + 'dataset/rsna-bone-age/boneage-test-dataset.csv',
                           image_dir = defualt_path + 'dataset/rsna-bone-age/boneage-test-dataset/boneage-test-dataset/',
                           basedOnSex = basedOnSex, gender = gender)
    
    num_classes = train_dataset.num_classes                   
    net = ResNet18(image_channels=1, num_classes=num_classes)
    logging.info(f'Network:\n'
                 f'\t{net.in_channels} input channels\n'
                 f'\t{net.num_classes} output channels (classes)\n')
    
    learning_rate = 0.001
    epochs = 10
    batch_size = 2
    val_percent = 0.1
    train_loader, val_loader, test_loader = data_organizer(train_dataset, test_dataset, 
                                    batch_size, val_percent = val_percent, shuffle = False, num_workers = 1)

    # train_net(net, device, train_loader, val_loader, val_percent
    #         epochs, batch_size, learning_rate)
    run_name = time_now.strftime("%Y%m%d_%H%M%S") + f"_{net.name}"
    try:
        train_net(net = net,
                  device = device,
                  train_loader = train_loader, 
                  val_loader = val_loader, 
                  epochs = epochs, 
                  batch_size = batch_size, 
                  learning_rate = learning_rate, 
                  val_percent = val_percent,
                  amp = False, 
                  save_checkpoint = True, 
                  dir_checkpoint = './checkpoints/',
                  run_name = run_name)
    except KeyboardInterrupt:
        torch.save(net.state_dict(), './checkpoints/INTERRUPTED.pth')
        logging.info('Saved interrupt')
        sys.exit(0)
