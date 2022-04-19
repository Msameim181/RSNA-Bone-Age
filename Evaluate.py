# System and utils for preprocessing
import logging

# Deep learning libs
import torch
from tqdm import tqdm


def evaluate(net, test_loader, device, criterion):
    """
    Validate the model on the validation set
    """
    net.eval()
    n_val = len(test_loader.dataset)
    test_loss = 0
    correct = 0

    for _, images, gender in tqdm(test_loader, total = n_val, desc='Validation Round...', unit = 'img', leave=False):

        images = torch.unsqueeze(images, 1)
        images = images.to(device = device, dtype = torch.float32)
        gender = gender.to(device = device, dtype = torch.float32)
        # boneage_onehot = torch.nn.functional.one_hot(torch.tensor(boneage), num_classes = int(num_classes))
        # target_age = boneage_onehot.to(device = device, dtype = torch.float32)
        # t_age = boneage.to(device = device, dtype = torch.float32)

        # with torch.no_grad():
        #     output_age = net([images, gender])
        #     val_loss += criterion(output_age, target_age)  # sum up batch loss
        #     # val_loss += torch.nn.functional.cross_entropy(output_age, target_age, reduction='sum').item()  # sum up batch loss
        #     pred = output_age.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        #     correct += pred.eq(t_age.view_as(pred)).sum().item()
    
    # acc = correct
    # if n_val != 0:
    #     val_loss /= n_val
    #     acc /= n_val

    # # Logging
    # logging.info(f'\nValidation set:\n'
    #              f'\tAverage loss: {val_loss:.4f}'
    #              f'\tAccuracy: {acc * 100:.2f}% \t Correct = {correct}/{n_val}\n')
    
    # return val_loss, acc, correct
