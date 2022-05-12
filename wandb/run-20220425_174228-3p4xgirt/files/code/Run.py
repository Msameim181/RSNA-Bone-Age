# System and utils for preprocessing
import logging
import sys
from datetime import datetime
from pathlib import Path

# Deep learning libs
import torch

# Models
from models.MobileNet import MobileNet_V2
from models.ResNet import ResNet18, ResNet34, ResNet50, ResNet101, ResNet152
# Custom libs
from Train import trainer
from utils.dataloader import RSNATestDataset, RSNATrainDataset, data_wrapper


if __name__ == '__main__':

    # Set up logger
    logging.basicConfig(level = logging.INFO, format = '%(levelname)s: %(message)s')
    logging.info('Starting program...')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info('Device recognized.')
    logging.info(f'Using device "{device}", Now!')

    # Cuda free memory
    torch.cuda.empty_cache()
    torch.cuda.memory_summary(device = None, abbreviated = False)
    logging.info('Cuda memory released.')

    # Load data
    logging.info('Loading data...')
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

    # Loading NN model
    logging.info('Loading NN Model...')
    net = ResNet34(pretrained = True, image_channels=1, num_classes=num_classes)
    logging.info(f'Model loaded as "{net.name}"')
    logging.info(f'Network:\n'
                 f'\t{net.in_channels} input channels\n'
                 f'\t{net.num_classes} output channels (classes)\n')

    # Set up training hyperparameters
    logging.info('Reading hyperparameters...')
    learning_rate = 0.001
    epochs = 10
    batch_size = 2
    val_percent = 0.3
    WandB_usage = True

    # Packaging the data
    logging.info('Packaging the data...')
    train_loader, val_loader, test_loader = data_wrapper(
                                                train_dataset, 
                                                test_dataset, 
                                                batch_size, 
                                                val_percent = val_percent, 
                                                shuffle = False, 
                                                num_workers = 1)
    logging.info(f'''Data Packaged as:
        Training Batch Size:    {batch_size}
        Training Size:          {len(train_loader.dataset)}
        Validation Size:        {len(val_loader.dataset)}
        Test Size:              {len(test_loader.dataset)}
        Validation %:           {val_percent}
    ''')

    # Set up Time
    time_now = datetime.now()
    run_name = time_now.strftime("%Y%m%d_%H%M%S") + f"_{net.name}"
    logging.info(f'Program started at {time_now}, Setup completed.')
    logging.info("Initiating training phase...")

    # Initiate training
    trainer(
        net = net,
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
        run_name = run_name,
        WandB_usage = WandB_usage)

    