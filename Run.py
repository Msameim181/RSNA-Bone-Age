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
    torch.cuda.memory_summary(device = None, abbreviated = False)
    rich_print('[INFO]: Cuda memory released.')

    # Load data
    rich_print('\n[INFO]: Loading data...')
    dataset_name = "rsna-bone-age" if args.dataset == "rsna" else "rsna-bone-age-kaggle" # rsna-bone-age-kaggle or rsna-bone-age
    basedOnSex = args.basedOnSex
    gender = 'male' if args.gender == 'male' else 'female'
    vars(args)['dataset_name'] = dataset_name
    vars(args)['gender'] = gender

    rich_print(f'[INFO]: DataSet: <{dataset_name}>\n'
                f'\tBased On Gender: {basedOnSex}\n'
                f'\tTargeted Gender: "{gender}"')
    
    train_dataset , test_dataset = data_handler(dataset_name = dataset_name, defualt_path = '', 
                                        basedOnSex = basedOnSex, gender = gender, 
                                        transform_action = 'train', target_type = data_type_interpretor(args.output_type))
    num_classes = train_dataset.num_classes if args.num_classes == 0 else args.num_classes
    vars(args)['train_dataset_size'] = len(train_dataset)
    vars(args)['test_dataset_size'] = len(test_dataset)
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
    train_loader, val_loader, test_loader = data_wrapper(train_dataset = train_dataset, 
                                                test_dataset = test_dataset, 
                                                batch_size = batch_size, test_val_batch_size = 1,
                                                val_percent = val_percent, 
                                                shuffle = False, num_workers = 1)
    rich_print(f'''[INFO]: Data Packaged as:
        Training Batch Size:    {batch_size}
        Training Size:          {len(train_loader.dataset)}
        Validation Size:        {len(val_loader.dataset)}
        Test Size:              {len(test_loader.dataset)}
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
        train_loader = train_loader, 
        val_loader = val_loader, 
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
    evaluate(net, args, test_loader, device, criterion)

    rich_print(f'\n[INFO]: Shutting Down...')

    