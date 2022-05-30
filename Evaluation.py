# System and utils for preprocessing
import logging

# Deep learning libs
import torch
from tqdm import tqdm
from utils.optimize_loss import *
from utils.rich_logger import *
from utils.dataloader import *


def evaluate(
        net, 
        args, 
        test_loader: DataLoader, 
        device, 
        criterion, 
        log_results: bool = True):
    """
    Evaluation the model on the Evaluation set
    """
    net.eval()
    n_eval = len(test_loader.dataset)
    test_loss_first = 0
    test_loss_second = 0
    correct = 0
    
    for _, images, gender, target, boneage, ba_minmax, ba_zscore, boneage_onehot, _ in tqdm(test_loader, total = n_eval, desc='Evaluation Round...', unit = 'img', leave=False):

        images = images.to(device = device, dtype = torch.float32)

        gender = torch.unsqueeze(gender, 1)
        gender = gender.to(device = device, dtype = torch.float32)

        target_age = target.to(device = device, dtype = torch.float32)
        boneage = boneage.to(device = device, dtype = torch.float32)
        ba_minmax = ba_minmax.to(device = device, dtype = torch.float32)

        with torch.no_grad():
            age_pred = net([images, gender])

            test_loss_first += criterion(age_pred, target_age)  # sum up batch loss
            
            pred = test_loader.dataset.predict_compiler(age_pred)
            correct += pred.eq(boneage.view_as(pred)).sum().item()
            
            # test_loss_second += torch.nn.functional.mse_loss(pred, boneage.view_as(pred))
            test_loss_second += torch.nn.functional.mse_loss(test_loader.dataset.out_min_max_normal(pred.view_as(boneage)), ba_minmax)
    
    accuracy = correct
    if n_eval != 0:
        test_loss_first /= n_eval
        test_loss_second /= n_eval
        accuracy /= n_eval

    # Logging
    # Wandb and TB logger here
    if log_results:
        print("\n")
        rich_print(f'\n[INFO]: Evaluation set:\n'
                f'\tAverage loss (criterion): {test_loss_first:.4f} \t Average loss (MSE): {test_loss_second:.4f}\n'
                f'\tAccuracy: {accuracy * 100:.2f}% \t Correct = {correct}/{n_eval}\n')
    
        rich_print('\n[INFO]: Finished Testing Round.')

    return test_loss_first, test_loss_second, accuracy, correct




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

    _, _, test_loader = data_wrapper(train_dataset = train_dataset, 
                            test_dataset = test_dataset, 
                            batch_size = 1,
                            test_val_batch_size = 1, 
                            shuffle = False, num_workers = 1)
    
    # Select and import Model
    net = MobileNet_V3(pretrained = True, image_channels = 1, num_classes = train_dataset.num_classes).cuda()
    reload_model(net, "./ResultModels/20220523_110557_MobileNetV3_Pre/checkpoint_epoch25.pth")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    criterion = loss_funcion('bce_wl')
    
    print(evaluate(net, None, test_loader, device, criterion))
