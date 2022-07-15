# System and utils for preprocessing
import logging
import os
import sys
from datetime import datetime
from time import sleep

# Deep learning libs
import torch

# Custom libs
from Evaluation import evaluate
from Train import trainer
from utils.config_model import *
from utils.dataloader import *
from utils.get_args import get_args
from utils.rich_logger import *

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

if __name__ == '__main__':
    # Get args
    args = get_args()
    # Set up logger
    logging.basicConfig(level = logging.INFO, format = '%(levelname)s: %(message)s')
    rich_print('[INFO]: Starting program...')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    rich_print('[INFO]: Device recognized.')
    rich_print(f'[INFO]: Using device "{device}", Now!')

    # Cuda free memory
    torch.cuda.empty_cache()
    torch.cuda.memory_summary(device = device, abbreviated = False)
    rich_print('[INFO]: Cuda memory released.')

    # Load data
    rich_print('\n[INFO]: Loading data...')
    # rsna-bone-age-kaggle or rsna-bone-age or rsna-bone-age-neu
    dataset_keys = ['rsna', 'rsnae', 'rsnak']
    dataset_names = ['rsna-bone-age', 'rsna-bone-age-neu', 'rsna-bone-age-kaggle']
    dataset_name = dataset_names[dataset_keys.index(args.dataset)] if args.dataset in dataset_keys else "rsna-bone-age-kaggle"
    # dataset_name = "rsna-bone-age" if args.dataset == "rsna" else "rsna-bone-age-kaggle" # rsna-bone-age-kaggle or rsna-bone-age
    basedOnSex = args.basedOnSex
    gender = 'male' if args.gender == 'male' else 'female'
    vars(args)['dataset_name'] = dataset_name
    vars(args)['gender'] = gender

    rich_print(f'[INFO]: DataSet: <{dataset_name}>\n'
                f'\tBased On Gender: {basedOnSex}\n'
                f'\tTargeted Gender: "{gender}"')
    
    datasets = data_handler(dataset_name = dataset_name, defualt_path = '', 
        basedOnSex = basedOnSex, gender = gender, 
        transform_action = 'train', target_type = data_type_interpretor(args.output_type),
        age_filter = False, age_bound_selection = 1)
    num_classes = datasets['train_dataset'].num_classes if args.num_classes == 0 else args.num_classes
    vars(args)['train_dataset_size'] = len(datasets['train_dataset'])
    vars(args)['test_dataset_size'] = len(datasets['test_dataset'])
    vars(args)['target_type'] = data_type_interpretor(args.output_type)

    # Set up training hyperparameters
    rich_print('\n[INFO]: Reading hyperparameters...')
    learning_rate = args.learning_rate
    epochs = args.epochs
    batch_size = args.batch_size
    val_percent = args.val_percent
    WandB_usage = args.wandb == 'True'

    # Packaging the data
    rich_print('\n[INFO]: Packaging the data...')
    loaders = data_wrapper(train_dataset = datasets['train_dataset'], 
        test_dataset = datasets['test_dataset'], 
        batch_size = batch_size, test_val_batch_size = 1,
        val_percent = val_percent, 
        shuffle = False, num_workers = 1)
    rich_print(f'''[INFO]: Data Packaged as:
        Training Batch Size:    {batch_size}
        Training Size:          {len(loaders['train_loader'].dataset)}
        Validation Size:        {len(loaders['val_loader'].dataset)}
        Test Size:              {len(loaders['test_loader'].dataset)}
        Validation %:           {val_percent}
    ''')


    # Loading NN model
    rich_print('\n[INFO]: Loading NN Model...')
    name_suffix = f"_{gender}" if basedOnSex else ''
    net = select_model(args = args, image_channels=1, num_classes=num_classes, name_suffix = name_suffix)
    rich_print(f'[INFO]: Model loaded as <{net.name}>')
    rich_print(f'[INFO]: Network:\n'
                 f'\t{net.in_channels} input channels\n'
                 f'\t{net.num_classes} output channels (classes)')
    
    # Set up Time and run name
    start_time = datetime.now()
    name_suffix = f"_{args.name_suffix}" if args.name_suffix else ''
    run_name = start_time.strftime("%Y%m%d_%H%M%S") + f"_{net.name}" + name_suffix
    rich_print(f'[INFO]: Program started at {start_time.strftime("%Y-%m-%d %H:%M:%S")}, Setup completed.')
    rich_print("\n[INFO]: Initiating training phase...")

    # Initiate training
    criterion, wandb_logger, tb_logger = trainer(
        net = net,
        args = args,
        device = device,
        train_loader = loaders['train_loader'], 
        val_loader = loaders['val_loader'], 
        epochs = epochs, 
        batch_size = batch_size, 
        learning_rate = learning_rate, 
        val_percent = val_percent,
        amp = args.amp, 
        save_checkpoint = args.checkpoint == 'True', 
        dir_checkpoint = './checkpoints/',
        run_name = run_name,
        WandB_usage = WandB_usage,
        dataset_name = dataset_name,
        notes = args.notes,
    )

    # Ending Training
    end_time = datetime.now()
    rich_print(f'\n[INFO]: Finished Training Phase at {end_time.strftime("%Y-%m-%d %H:%M:%S")}.')
    rich_print(f'\n[INFO]: Training Duration: {end_time - start_time}.')

    rich_print("\n[INFO]: Initiating testing phase...")
    # Initiate testing
    _, _, _, _, _ = evaluate(
        net = net, 
        args = args, 
        test_loader = loaders['test_loader'], 
        device = device, 
        criterion = criterion, 
        WandB_usage = WandB_usage, 
        wandb_logger = wandb_logger, 
        tb_logger = tb_logger, 
        log_results = True, 
        logger_usage = True)

    # Ending Testing and Cleaning up the loggers
    tb_logger.close()

    rich_print(f'\n[INFO]: Shutting Down...')

    