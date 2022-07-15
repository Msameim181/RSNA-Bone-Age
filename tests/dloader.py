# OS and Path to main files
import os
import sys

p = os.path.abspath('.')
sys.path.insert(1, p)

# Standard Libs
import argparse

import matplotlib.pyplot as plt
import numpy as np
# Deep learning libs
import torch
# Main Runner
from Evaluation import evaluate
from PIL import Image
from tqdm import tqdm
# Utils
from utils.config_model import *
from utils.dataloader import *
from utils.optimize_loss import *
from utils.rich_logger import *
from utils.tensorboard_logger import *
from utils.wandb_logger import *


# Test
def plot_data(img, r, c, i):
    plt.subplot(r, c, i)
    plt.imshow(img, cmap='gray')
    plt.axis('off')
    plt.tight_layout()


if __name__ == '__main__':

    datasets = data_handler(dataset_name = 'rsna-bone-age', defualt_path = '', 
        basedOnSex = False, gender = 'male', target_type = 'minmax', 
        age_filter = True, age_bound_selection = 1)

    batch_size, val_percent = 1, 0.2
    loaders = data_wrapper(
        train_dataset = datasets['train_dataset'], 
        test_dataset = datasets['test_dataset'], 
        batch_size = batch_size, test_val_batch_size = 1,
        val_percent = val_percent, 
        shuffle = False, 
        num_workers = 1)

    print("Train: ", len(loaders['train_loader'].dataset))
    print("Valid: ", len(loaders['val_loader'].dataset))
    print("Tests:", len(loaders['test_loader'].dataset))

    
    # print(train_dataset.train_data_filtered['boneage'].count())
    # print(test_dataset.test_data_filtered['boneage'].count())


    # transf = torchvision.transforms.ToPILImage()
    count = 1
    # row = 5
    # col = 5
    train_loader = loaders['train_loader']
    test_loader = loaders['test_loader']
    with data_progress:
        
        for train_batch in data_progress.track(train_loader):
            img_id = train_batch['img_id']
            images = train_batch['image']
            gender = train_batch['gender']
            target = train_batch['target']
            boneage = train_batch['boneage']
            ba_minmax = train_batch['ba_minmax']
            ba_zscore = train_batch['ba_zscore']
            boneage_onehot = train_batch['boneage_onehot'], 
            num_classes = train_batch['num_classes']
            print(img_id, images.shape, gender, boneage.shape, ba_minmax.shape, ba_zscore)
            # print(train_loader.dataset.dataset.reverse_min_max_normal(ba_minmax), train_loader.dataset.dataset.a_min, train_loader.dataset.dataset.a_max)
            # print(target, train_loader.dataset.dataset.predict_compiler(target), train_loader.dataset.dataset.reverse_zscore_normal(ba_zscore))
            # print(train_loader.dataset.dataset.train_data)

            # img1 = transf(img[0])
            # plot_data(img1, row, col, count)
            # img1.save(f'zzz/{img_id.item()}.jpg')
            # # plot_data(img[0], row, col, count)

            count += 1
            if count == 25:
                break
            # break

        # plt.tight_layout()
        # plt.savefig("Gamma.png", dpi=300)
        # plt.show()

        # print("---------------")
    
        for img_id, img, sex, target, boneage, ba_minmax, ba_zscore, boneage_onehot, _ in data_progress.track(test_loader):
            print(img_id, img.shape, sex, boneage, ba_minmax, ba_zscore)
        #     print(test_loader.dataset.reverse_min_max_normal(ba_minmax), test_loader.dataset.a_min, test_loader.dataset.a_max)
        #     print(target, test_loader.dataset.predict_compiler(target), test_loader.dataset.reverse_zscore_normal(ba_zscore))
        #     print(test_loader.dataset.test_data)
            
            break

