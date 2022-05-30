# System and utils for preprocessing
import logging

# Deep learning libs
import torch
from tqdm import tqdm
from utils.optimize_loss import *
from utils.rich_logger import *
from utils.dataloader import *

def validate(
        net, 
        args, 
        val_loader: DataLoader, 
        device, 
        criterion,
        log_results: bool = True):
    """
    Validate the model on the validation set
    """
    net.eval()
    n_val = len(val_loader.dataset)
    val_loss = 0.0
    correct = 0


    for _, images, gender, target, boneage, ba_minmax, ba_zscore, boneage_onehot, _ in tqdm(val_loader, total = n_val, desc='Validation Round...', unit = 'img', leave=False):

        images = images.to(device = device, dtype = torch.float32)

        gender = torch.unsqueeze(gender, 1)
        gender = gender.to(device = device, dtype = torch.float32)

        target_age = target.to(device = device, dtype = torch.float32)
        boneage = boneage.to(device = device, dtype = torch.float32)

        with torch.no_grad():
            if args.basedOnSex and args.input_size == 1:
                age_pred = net(images)
            else:
                age_pred = net([images, gender])
            val_loss += criterion(age_pred, target_age)
            
            pred = val_loader.dataset.dataset.predict_compiler(age_pred) 

            correct += pred.eq(boneage.view_as(pred)).sum().item()
    
    accuracy = correct
    if n_val != 0:
        val_loss /= n_val
        accuracy /= n_val

    # Logging
    if log_results:
        print("\n")
        rich_print(f'\n[INFO]: Validation set:\n'
                f'\tAverage loss: {val_loss:.4f}\n'
                f'\tAccuracy: {accuracy * 100:.2f}% \t Correct = {correct}/{n_val}\n')
    
    return val_loss, accuracy, correct



# Testing
if __name__=='__main__':
    from utils.config_model import *
    from utils.dataloader import *
    dataset_name = "rsna-bone-age" # rsna-bone-age-kaggle or rsna-bone-age
    basedOnSex = False
    gender='male'

    train_dataset , test_dataset = data_handler(dataset_name = dataset_name, defualt_path = '', 
                                        basedOnSex = basedOnSex, gender = gender, transform_action = 'train', target_type = 'onehot')
    num_classes = train_dataset.num_classes 

    _, val_loader, _ = data_wrapper(train_dataset = train_dataset, 
                            test_dataset = test_dataset, 
                            batch_size = 1,
                            test_val_batch_size = 1, 
                            shuffle = False, num_workers = 1)
    
    # Select and import Model
    net = MobileNet_V3(pretrained = True, image_channels = 1, num_classes = train_dataset.num_classes).cuda()
    reload_model(net, "./ResultModels/20220523_110557_MobileNetV3_Pre/checkpoint_epoch25.pth")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    criterion = loss_funcion('bce_wl')


    import argparse
    parser = argparse.ArgumentParser(description='Train the Your Model on images and target age.')
    args = parser.parse_args()
    vars(args)['basedOnSex'] = False
    vars(args)['input_size'] = 2

    print(validate(net, args, val_loader, device, criterion))
