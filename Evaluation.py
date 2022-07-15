# System and utils for preprocessing
import argparse

# Deep learning libs
import torch
from tqdm import tqdm

from utils.dataloader import *
from utils.optimize_loss import *
from utils.rich_logger import *
from utils.tensorboard_logger import *
from utils.wandb_logger import *


# Utils for preprocessing
def log_evaluation_results(WandB_usage: bool, wandb_logger, tb_logger, result):
    """Logs the evaluation results."""
    if WandB_usage:
        wandb_log_evaluation(wandb_logger, result)

    tb_log_evaluation(tb_logger, result)

def find_best_and_worst(idx_img, true_ages, predictions, num: int = 5):
    """Finds the top 5 best and worst prediction.
    
    Args:
        idx_img (list): List of image indices
        true_ages (list): List of true ages
        predictions (list): List of predictions
        num (int): Number of top predictions to find

    Returns:
        dict: Top 5 predictions with their image, difference, true age and prediction (Best and Worst)
    """
    # true_ages, predictions, idx_img
    diffs = [abs(true_age - pred) for idx, true_age, pred in zip(idx_img, true_ages, predictions)]

    # Sort
    diffs, true_ages, predictions, idx_img = (list(t) for t in zip(*sorted(zip(diffs, true_ages, predictions, idx_img), key=lambda pair: pair[0]))) 

    best_predictions = dict(
        difference = diffs[:num],
        true_age = true_ages[:num],
        prediction = predictions[:num],
        idx_img = idx_img[:num])
    worst_predictions = dict(
        difference = diffs[-num:],
        true_age = true_ages[-num:],
        prediction = predictions[-num:],
        idx_img = idx_img[-num:])
    
    return best_predictions, worst_predictions

def add_info_to_images(predictions: dict):
    """Adds information to the predicted images as a text.

    Args:
        predictions (dict): Top 5 predictions with their image, difference, true age and prediction (Best and Worst)

    Returns:
        dict: Top 5 predictions with their image, difference, true age and prediction (Best and Worst)
    """
    for item in range(len(predictions['difference'])):
        text = (f"Id: {predictions['idx_img'][item]}\n"
        f"Max Diff: {predictions['difference'][item]:.4f}\n"
        f"True Age: {predictions['true_age'][item]:.1f}\n"
        f"Pred Age: {predictions['prediction'][item]:.1f}\n")
        predictions['predictions_images'][item] = add_text_to_image(predictions['predictions_images'][item], text)
    predictions['predictions_images'] = np.array(predictions['predictions_images'])
    return predictions


def evaluate(
        net, 
        args, 
        test_loader: DataLoader, 
        device, 
        criterion, 
        WandB_usage: bool, 
        wandb_logger = None, 
        tb_logger = None, 
        log_results: bool = True,
        logger_usage: bool = True):
    """
    Evaluation the model on the Evaluation set
    """
    net.eval()
    n_eval = len(test_loader.dataset)
    test_loss_first = 0
    test_loss_second = 0
    test_loss_third = 0
    test_loss_mse_age = 0
    test_loss_mae_age = 0
    correct = 0
    predictions = []
    true_ages = []
    idx_img = []

    for eval_batch in tqdm(test_loader, total = n_eval, desc='Evaluation Round...', unit = 'img', leave=True):
        
        # Unpacking the data
        idx = eval_batch['img_id']
        images = eval_batch['image']
        gender = eval_batch['gender']
        target = eval_batch['target']
        boneage = eval_batch['boneage']
        ba_minmax = eval_batch['ba_minmax']
        
        # Processing data before feeding the network
        images = images.to(device = device, dtype = torch.float32)

        gender = torch.unsqueeze(gender, 1)
        gender = gender.to(device = device, dtype = torch.float32)

        target = target.to(device = device, dtype = torch.float32)
        boneage = boneage.to(device = device, dtype = torch.float32)
        ba_minmax = ba_minmax.to(device = device, dtype = torch.float32)

        with torch.no_grad():
            if args.basedOnSex and args.input_size == 1:
                if args.attention:
                    age_pred, _, _, _ = net(images)
                else:
                    age_pred = net(images)
            elif args.attention:
                age_pred, _, _, _ = net([images, gender])
            else:
                age_pred = net([images, gender])

            # Compiling real predict ages based on dataloader
            pred = test_loader.dataset.predict_compiler(age_pred)

            # Storing results for future use: predictions, true_ages, idx_img
            predictions.append(pred.cpu().numpy().item())
            true_ages.append(boneage.cpu().numpy().item())
            idx_img.append(idx.cpu().numpy().item())

            # Calculating correction: The number of correct result on onehot encoded boneage
            correct += pred.eq(boneage.view_as(pred)).sum().item()

            # Calculating losses
            # Loss with main criterion on normal target ages
            test_loss_first += criterion(age_pred, target.view_as(age_pred))
            # Loss with second MSE & MAE on normal ages min_max as we want
            test_loss_second += torch.nn.functional.mse_loss(test_loader.dataset.out_min_max_normal(pred.view_as(boneage)), ba_minmax)
            test_loss_third += torch.nn.functional.l1_loss(test_loader.dataset.out_min_max_normal(pred.view_as(boneage)), ba_minmax)
            # Loss with MSE & MAE on true ages
            test_loss_mse_age += torch.nn.functional.mse_loss(pred.view_as(boneage), boneage).float()
            test_loss_mae_age += torch.nn.functional.l1_loss(pred.view_as(boneage), boneage).float()

    # Calculating the mean of the losses and the accuracy
    accuracy = correct
    if n_eval != 0:
        test_loss_first /= n_eval
        test_loss_second /= n_eval
        test_loss_third /= n_eval
        test_loss_mse_age /= n_eval
        test_loss_mae_age /= n_eval
        accuracy /= n_eval
    
    # Sort based on boneage for charts
    true_ages, predictions, idx_img = (list(t) for t in zip(*sorted(zip(true_ages, predictions, idx_img), key=lambda pair: pair[0])))

    # Find best and worst predictions
    best_predictions, worst_predictions = find_best_and_worst(idx_img, true_ages, predictions)

    # Loading the best and worst predictions images
    best_predictions['predictions_images'] = [test_loader.dataset.load_image(idx) for idx in best_predictions['idx_img']]
    worst_predictions['predictions_images'] = [test_loader.dataset.load_image(idx) for idx in worst_predictions['idx_img']]
    # Add information (text) to the images
    best_predictions, worst_predictions = add_info_to_images(best_predictions), add_info_to_images(worst_predictions)

    # Logging
    # Wandb and TB logger here
    if log_results:
        print("\n")
        rich_print(f'\n[INFO]: Evaluation set:\n'
                f'\tAverage loss (criterion): {test_loss_first:.10f}\n'
                f'\t Accuracy: {accuracy * 100:.2f}% \t\t Correct = {correct}/{n_eval}\n'
                f'\t MSE loss "MinMax": {test_loss_second:.8f} \t MAE loss "MinMax": {test_loss_third:.8f}\n'
                f'\t MSE loss Age(m): {test_loss_mse_age:.4f} \t MAE loss Age(m): {test_loss_mae_age:.4f}\n')
    
        rich_print('\n[INFO]: Finished Testing Round.')

    if logger_usage or tb_logger != None:
        result = dict(
            test_loss_first = test_loss_first,
            test_loss_second = test_loss_second,
            test_loss_third = test_loss_third,
            test_loss_mse_age = test_loss_mse_age,
            test_loss_mae_age = test_loss_mae_age,
            accuracy = accuracy,
            correct = correct,
            n_eval = n_eval,
            boneage = np.array(true_ages),
            pred = np.array(predictions),
            best_predictions = best_predictions,
            worst_predictions = worst_predictions,
        )
        log_evaluation_results(WandB_usage, wandb_logger, tb_logger, result)
    
    return test_loss_first, accuracy, correct, best_predictions, worst_predictions
