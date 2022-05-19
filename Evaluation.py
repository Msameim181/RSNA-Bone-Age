# System and utils for preprocessing
import logging

# Deep learning libs
import torch
from tqdm import tqdm

from utils.rich_logger import make_console

console = make_console()

def evaluate(net, test_loader, device, criterion):
    """
    Evaluation the model on the Evaluation set
    """
    net.eval()
    n_eval = len(test_loader.dataset)
    test_loss_first = 0
    test_loss_second = 0
    correct = 0

    for _, images, boneage, boneage_onehot, gender in tqdm(test_loader, total = n_eval, desc='Evaluation Round...', unit = 'img', leave=False):

        images = torch.unsqueeze(images, 1)
        images = images.to(device = device, dtype = torch.float32)

        gender = gender.to(device = device, dtype = torch.float32)

        # boneage_onehot = torch.nn.functional.one_hot(torch.tensor(boneage), num_classes = int(num_classes))
        target_age = boneage_onehot.to(device = device, dtype = torch.float32)
        t_age = boneage.to(device = device, dtype = torch.float32)

        with torch.no_grad():
            output_age = net([images, gender])
            test_loss_first += criterion(output_age, target_age)  # sum up batch loss
            
            # test_loss += torch.nn.functional.cross_entropy(output_age, target_age, reduction='sum').item()  # sum up batch loss
            pred = output_age.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(t_age.view_as(pred)).sum().item()
            
            test_loss_second += torch.nn.functional.mse_loss(pred, t_age.view_as(pred))
    
    acc = correct
    if n_eval != 0:
        test_loss_first /= n_eval
        test_loss_second /= n_eval
        acc /= n_eval

    # Logging
    console.print(f'\n[INFO]: Evaluation set:\n'
                 f'\tAverage loss (criterion): {test_loss_first:.4f}'
                 f'\tAverage loss (MSE): {test_loss_second:.4f}'
                 f'\tAccuracy: {acc * 100:.2f}% \t Correct = {correct}/{n_eval}\n')
    
    return test_loss_first, test_loss_second, acc, correct
