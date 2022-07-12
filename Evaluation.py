# System and utils for preprocessing
import argparse
import logging

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

    for idx, images, gender, target, boneage, ba_minmax, ba_zscore, boneage_onehot, _ in tqdm(test_loader, total = n_eval, desc='Evaluation Round...', unit = 'img', leave=False):

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
    
    return best_predictions, worst_predictions



# Testing

def make_fake_args():
    args = argparse.ArgumentParser()
    return args.parse_args()

if __name__=='__main__':
    from utils.config_model import *
    from utils.dataloader import *
    dataset_name = "rsna-bone-age" # rsna-bone-age-kaggle or rsna-bone-age
    basedOnSex = False
    gender='male'
    args = make_fake_args()
    vars(args)['basedOnSex'] = False
    vars(args)['attention'] = False

    train_dataset , test_dataset = data_handler(dataset_name = dataset_name, defualt_path = '', 
                                        basedOnSex = basedOnSex, gender = gender, transform_action = 'train', target_type = 'minmax')
    num_classes = train_dataset.num_classes 

    _, _, test_loader = data_wrapper(train_dataset = train_dataset, 
                            test_dataset = test_dataset, 
                            batch_size = 1,
                            test_val_batch_size = 1, 
                            shuffle = False, num_workers = 1)
    
    # Select and import Model
    # net = MobileNet_V3(pretrained = True, image_channels = 1, num_classes = train_dataset.num_classes).cuda()

    net = load_model("./ResultModels/20220628_111939_MobileNetV3_Pre_MSE_G-32/checkpoint_model.pth").cuda()
    # reload_model(net, "./ResultModels/20220619_172133_MobileNetV3_Pre_MSE_G-FC32_RSNA/checkpoint_epoch17.pth")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    criterion = loss_funcion('mse')
    # test_loss_first, test_loss_second, accuracy, correct, true_ages, predictions, idx_img = evaluate(net, args, test_loader, device, criterion, 
    best_predictions, worst_predictions = evaluate(net, args, test_loader, device, criterion, 
            logger_usage = False, WandB_usage = False, tb_logger = None, wandb_logger = None)
    # print(test_loss_first, test_loss_second, accuracy, correct)
    # print("-----------------------------------------------------")
    # print(true_ages, predictions)
    # print("-----------------------------------------------------")

    # import matplotlib.pyplot as plt
    
    # plt.plot(np.array(true_ages), 'r', label = 'True')
    # plt.plot(np.array(predictions), 'b', label = 'Pred')
    # plt.legend()
    # plt.show()

    # --------------------------------------------

    # best_predictions, worst_predictions = find_best_and_worst(idx_img, true_ages, predictions)
    # print(best_predictions)
    # print("---------------------")
    # print(worst_predictions)
    # print("---------------------")

    # for item in range(len(best_predictions['difference'])):
    #     print(f"{best_predictions['idx_img'][item]}: {best_predictions['difference'][item]:.4f} / {best_predictions['true_age'][item]:.4f} / {best_predictions['prediction'][item]:.4f}")
    # print("---------------------")

    # for item in range(len(worst_predictions['difference'])):
    #     print(f"{worst_predictions['idx_img'][item]}: {worst_predictions['difference'][item]:.4f} / {worst_predictions['true_age'][item]:.4f} / {worst_predictions['prediction'][item]:.4f}")
    # print("---------------------")


    # for item in range(len(best_predictions['difference'])):
    #     img = Image.fromarray(best_predictions['predictions_images'][item])

    #     img.show()

    # for item in range(len(worst_predictions['difference'])):
    #     img = Image.fromarray(worst_predictions['predictions_images'][item])

    #     img.show()

    tb_logger = tb_rewrite_log('tensorboardLocal/Part2/20220628_111939_MobileNetV3_Pre_MSE_G-32')
    result = dict(
        best_predictions = best_predictions,
        worst_predictions = worst_predictions,
    )
    tb_log_evaluation_images(tb_logger, result)
