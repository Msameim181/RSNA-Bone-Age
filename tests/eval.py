# OS and Path to main files
import os
import sys

p = os.path.abspath('.')
sys.path.insert(1, p)

# Standard Libs
import argparse

import numpy as np

# Deep learning libs
import torch
# Main Runner
from Evaluation import evaluate
from PIL import Image
# Utils
from utils.config_model import *
from utils.dataloader import *
from utils.optimize_loss import *
from utils.rich_logger import *
from utils.tensorboard_logger import *
from utils.wandb_logger import *

# Testing

def sava_image_results(predictions: dict, save_dir: str, type: str):
    # create directory if it doesn't exist: "save_dir/type" using Path.mkdir(parents=True, exist_ok=True)
    Path(save_dir + "/" + type).mkdir(parents=True, exist_ok=True)
    
    
    for item in range(len(predictions['difference'])):
        img = predictions['predictions_images'][item]
        # Change the image to a PIL image and save it as a PNG
        img = Image.fromarray(img)
        img.save(os.path.join(save_dir, f'{type}/{type}_Prediction_' + str(item) + '.png'))
        
    


def make_fake_args():
    args = argparse.ArgumentParser()
    return args.parse_args()

if __name__=='__main__':
    dataset_name = "rsna-bone-age-neu" # rsna-bone-age-kaggle or rsna-bone-age
    basedOnSex = True
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

    net = load_model("./ResultModels/20220626_113847_MobileNetV3_Pre_male_MSE_G-32_Male/checkpoint_model.pth").cuda()
    # reload_model(net, "./ResultModels/20220619_172133_MobileNetV3_Pre_MSE_G-FC32_RSNA/checkpoint_epoch17.pth")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    criterion = loss_funcion('mse')
    # test_loss_first, test_loss_second, accuracy, correct, true_ages, predictions, idx_img = evaluate(net, args, test_loader, device, criterion, 
    test_loss_first, accuracy, correct, best_predictions, worst_predictions = evaluate(net, args, test_loader, device, criterion, 
            logger_usage = False, WandB_usage = False, tb_logger = None, wandb_logger = None)
    
    
    sava_image_results(predictions = best_predictions,
                       save_dir = 'tensorboardLocal/Part2/20220626_113847_MobileNetV3_Pre_male_MSE_G-32_Male/Top5', 
                       type = 'Best')
    sava_image_results(predictions = worst_predictions,
                       save_dir = 'tensorboardLocal/Part2/20220626_113847_MobileNetV3_Pre_male_MSE_G-32_Male/Top5', 
                       type = 'Worst')
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

    # tb_logger = tb_rewrite_log('tensorboardLocal/Part2/20220628_111939_MobileNetV3_Pre_MSE_G-32')
    # result = dict(
    #     best_predictions = best_predictions,
    #     worst_predictions = worst_predictions,
    # )
    # tb_log_evaluation_images(tb_logger, result)
