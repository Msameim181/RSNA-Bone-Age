# System and utils for preprocessing
import logging
import os
from pathlib import Path

import albumentations as A
import cv2
import matplotlib.pyplot as plt
# Deep learning libs
import numpy as np
import pandas as pd
import torch
import torchvision
from albumentations.pytorch.transforms import ToTensorV2
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split

from utils.rich_logger import *
# from rich_logger import *


# Defining the dataset class
class RSNATrainDataset(Dataset):
    def __init__(self, data_file: str, image_dir: str, 
            transform = None, basedOnSex: bool=False, 
            gender:str='male', target_type: str = 'onehot'):
        """Train dataset.

        Args:
            data_file (str): Path to the dataset file (.csv).
            image_dir (str): Path to the dataset image directory.
            transform (optional): The transformation on data. Defaults to None.
            basedOnSex (bool, optional): Data will be based on gender. Defaults to False.
            gender (str, optional): _description_. Defaults to 'male'. Options: ['male', 'female']
            target_type (str, optional): Data targeted. What type age data. Defaults to 'onehot'. 
                Options: ['onehot', 'minmax', 'zscore', 'real']

        Raises:
            RuntimeError: No data file found.
            RuntimeError: Train data is empty file.
        """
        self.data_file = Path(data_file)
        self.image_dir = Path(image_dir)
        self.transform = transform
        self.basedOnSex = basedOnSex
        self.gender = gender
        self.target_type = target_type

        # Proccessing data and csv file
        self.train_data = pd.read_csv(data_file)

        # Normalization Min, Max
        self.train_data['ba_minmax'], self.a_min, self.a_max = self.min_max_normal(self.train_data['boneage'].copy())
        # Normalization Zscore
        self.train_data['ba_zscore'], self.a_mean, self.a_std = self.zscore_normal(self.train_data['boneage'].copy())
        # Number of classes for the one hot encoding
        self.num_classes = np.max(self.train_data['boneage']) + 1
        # One Hoting the bone age
        self.age_onehot  = torch.nn.functional.one_hot(torch.tensor(self.train_data['boneage']), num_classes = self.num_classes)
        # Remebering the index for the one hot encoding and gender discrimination
        self.train_data['indx'] = range(len(self.train_data))

        # Dividing data based on gender (gender discrimination)
        if self.basedOnSex and self.gender == 'male':
            self.train_data_filtered = self.train_data[self.train_data['male'] == True]
        elif self.basedOnSex and self.gender == 'female':
            self.train_data_filtered = self.train_data[self.train_data['male'] == False]
        else:
            self.train_data_filtered = self.train_data

        # Raise Errors
        if not os.path.exists(data_file):
            raise RuntimeError(f'No data file found in {data_file}.')
        if self.train_data.empty:
            raise RuntimeError('Train data is empty file')

        rich_print(f"[INFO]: Creating 'Train' dataset with {len(self.train_data)} samples.")

    def __len__(self):
        return len(self.train_data_filtered)

    def __getitem__(self, index):
        # Image, first data
        img_id = self.train_data_filtered.iloc[index].id
        img_addr = Path(self.image_dir, f'{str(img_id)}.png')

        assert os.path.exists(img_addr), f'Image {img_addr} does not exist'

        # Gender, second data
        sex = 1 if self.train_data_filtered.iloc[index].male else 0

        # Age, Target data
        boneage = self.train_data_filtered.iloc[index].boneage
        ba_minmax = self.train_data_filtered.iloc[index].ba_minmax
        ba_zscore = self.train_data_filtered.iloc[index].ba_zscore
        onehot_index = self.train_data_filtered.iloc[index]['indx']
        boneage_onehot = self.age_onehot[onehot_index]
        
        # Find target
        if self.target_type == 'onehot':
            target = boneage_onehot
        elif self.target_type == 'minmax':
            target = ba_minmax
        elif self.target_type == 'zscore':
            target = ba_zscore
        elif self.target_type == 'real' or self.target_type:
            target = boneage

        # Number of classes for models output shape, if needed.
        num_classes = self.num_classes

        # Image preprocessing
        img = Image.open(img_addr)
        # img = img.resize((500, 625))
        img = np.array(img)
        
        if self.transform is not None:
            augmentations = self.transform(image=img)
            img = augmentations["image"]


        return img_id, img, sex, target, boneage, ba_minmax, ba_zscore, boneage_onehot, num_classes

    def min_max_normal(self, ba_minmax):
        a_min, a_max = ba_minmax.min(), ba_minmax.max()
        ba_minmax -= a_min
        ba_minmax /= a_max
        return ba_minmax, a_min, a_max

    def reverse_min_max_normal(self, item):
        return (item * self.a_max) + self.a_min

    def zscore_normal(self, ba_zscore):
        a_mean = ba_zscore.mean()
        a_std = ba_zscore.std()
        ba_zscore = (ba_zscore - ba_zscore.mean()) / ba_zscore.std()
        return ba_zscore, a_mean, a_std

    def reverse_zscore_normal(self, item):
        return (item * self.a_std) + self.a_mean
    
    def predict_compiler(self, preds):
        if self.target_type == 'onehot':
            return preds.argmax(dim=1, keepdim=True)
        elif self.target_type == 'minmax':
            return self.reverse_min_max_normal(preds)
        elif self.target_type == 'zscore':
            return self.reverse_zscore_normal(preds)
        elif self.target_type == 'real':
            return preds

class RSNATestDataset(Dataset):
    def __init__(self, data_file: str, image_dir: str, 
            train_num_classes: int, transform = None, 
            basedOnSex: bool = False, gender:str='male', 
            target_type: str = 'onehot'):
        
        """Test dataset.

        Args:
            data_file (str): Path to the dataset file (.csv).
            image_dir (str): Path to the dataset image directory.
            transform (optional): The transformation on data. Defaults to None.
            train_num_classes (int): Number of classes for the one hot encoding.
            basedOnSex (bool, optional): Data will be based on gender. Defaults to False.
            gender (str, optional): _description_. Defaults to 'male'. Options: ['male', 'female']
            target_type (str, optional): Data targeted. What type age data. Defaults to 'onehot'. 
                Options: ['onehot', 'minmax', 'zscore', 'real']

        Raises:
            RuntimeError: No data file found.
            RuntimeError: Test data is empty file.
        """


        self.data_file = Path(data_file)
        self.image_dir = Path(image_dir)
        self.transform = transform
        self.basedOnSex = basedOnSex
        self.gender = gender
        self.target_type = target_type

        # Proccessing data and csv file
        self.test_data = pd.read_csv(data_file)

        # Normalization Min, Max
        self.test_data['ba_minmax'], self.a_min, self.a_max = self.min_max_normal(self.test_data['boneage'].copy())
        # Normalization Zscore
        self.test_data['ba_zscore'], self.a_mean, self.a_std = self.zscore_normal(self.test_data['boneage'].copy())
        # One Hoting the bone age
        self.age_onehot  = torch.nn.functional.one_hot(torch.tensor(self.test_data['boneage']), num_classes = train_num_classes)
        # Remebering the index for the one hot encoding and gender discrimination
        self.test_data['indx'] = range(len(self.test_data))

        # Dividing data based on gender
        if self.basedOnSex and self.gender == 'male':
            self.test_data_filtered = self.test_data[self.test_data['male'] == True]
        elif self.basedOnSex and self.gender == 'female':
            self.test_data_filtered = self.test_data[self.test_data['male'] == False]
        else:
            self.test_data_filtered = self.test_data

        # Raise Errors
        if not os.path.exists(self.data_file):
            raise RuntimeError(f'No data file found in {data_file}.')
        if self.test_data.empty:
            raise RuntimeError('Test data is empty file')

        rich_print(f"[INFO]: Creating 'Test' dataset with {len(self.test_data)} samples.")

    def __len__(self):
        return len(self.test_data_filtered)

    def __getitem__(self, index):
        # Image, first data
        img_id = self.test_data_filtered.iloc[index].id
        img_addr = Path(self.image_dir, f'{str(img_id)}.png')

        assert os.path.exists(img_addr), f'Image {img_addr} does not exist'

        # Gender, second data
        sex = 1 if self.test_data_filtered.iloc[index].male else 0

        # Age, Target data
        boneage = self.test_data_filtered.iloc[index].boneage
        ba_minmax = self.test_data_filtered.iloc[index].ba_minmax
        ba_zscore = self.test_data_filtered.iloc[index].ba_zscore
        onehot_index = self.test_data_filtered.iloc[index]['indx']
        boneage_onehot = self.age_onehot[onehot_index]
        
        # Find target
        if self.target_type == 'onehot':
            target = boneage_onehot
        elif self.target_type == 'minmax':
            target = ba_minmax
        elif self.target_type == 'zscore':
            target = ba_zscore
        elif self.target_type == 'real' or self.target_type:
            target = boneage

        # Image preprocessing
        img = Image.open(img_addr)
        # img = img.resize((500, 625))
        img = np.array(img)

        if self.transform is not None:
            augmentations = self.transform(image=img)
            img = augmentations["image"]

        return img_id, img, sex, target, boneage, ba_minmax, ba_zscore, boneage_onehot, 0

    def min_max_normal(self, ba_minmax):
        a_min, a_max = ba_minmax.min(), ba_minmax.max()
        ba_minmax -= a_min
        ba_minmax /= a_max
        return ba_minmax, a_min, a_max
    
    def out_min_max_normal(self, item):
        return (item - self.a_min) / self.a_max

    def reverse_min_max_normal(self, item):
        return (item * self.a_max) + self.a_min

    def zscore_normal(self, ba_zscore):
        a_mean = ba_zscore.mean()
        a_std = ba_zscore.std()
        ba_zscore = (ba_zscore - ba_zscore.mean()) / ba_zscore.std()
        return ba_zscore, a_mean, a_std

    def reverse_zscore_normal(self, item):
        return (item * self.a_std) + self.a_mean
    
    def predict_compiler(self, preds):
        if self.target_type == 'onehot':
            return preds.argmax(dim=1, keepdim=True)
        elif self.target_type == 'minmax':
            return self.reverse_min_max_normal(preds)
        elif self.target_type == 'zscore':
            return self.reverse_zscore_normal(preds)
        elif self.target_type == 'real':
            return preds


def data_augmentation():
    return A.Compose([
        A.Resize(625, 500),
        A.ColorJitter(brightness=0.0, contrast=0.5, saturation=0.5, hue=0.5, p=0.9, always_apply=True),
        A.Rotate(limit=45, p=0.5),
        A.HorizontalFlip(p=0.2),
        ToTensorV2(),
    ])

def basic_transform():
    return A.Compose([
        A.Resize(625, 500),
        ToTensorV2(),
    ])

def data_type_interpretor(data_type):
    if data_type == 1:
        return 'onehot'
    elif data_type == 2:
        return 'minmax'
    elif data_type == 3:
        return 'zscore'
    else:
        return 'real'

# Data Packaging
def data_wrapper(train_dataset: Dataset, test_dataset: Dataset, 
        batch_size: int = 2, test_val_batch_size: int = 1, val_percent: float = 0.2, 
        shuffle: bool = True, num_workers: int = 1) -> DataLoader:
    """ Generate the train, validation and test dataloader for model.

    Args:
        train_dataset (dataset class): training dataset.
        test_dataset (dataset class): test dataset.
        batch_size (int): batch size.
        val_percent (float, optional): valifation percentage. Defaults to 0.2.
        shuffle (bool, optional): shuffle data. Defaults to True.
        num_workers (int, optional): number of worker. Defaults to 1.

    Returns:
        data loaders: train_loader, val_loader, test_loader
    """
    n_val = int(len(train_dataset) * val_percent)
    n_train = len(train_dataset) - n_val
    train_set, val_set = random_split(train_dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))
    train_loader = DataLoader(dataset = train_set, batch_size = batch_size, shuffle = shuffle, num_workers = num_workers, pin_memory = True)
    val_loader = DataLoader(dataset = val_set, batch_size = test_val_batch_size, shuffle = shuffle, num_workers = num_workers, pin_memory = True)

    test_loader = DataLoader(dataset = test_dataset, batch_size = test_val_batch_size, shuffle = shuffle, num_workers = num_workers, pin_memory = True)

    return train_loader, val_loader, test_loader

# Data Loading
def data_handler(dataset_name:str = 'rsna-bone-age-kaggle', defualt_path: str = '', 
        basedOnSex: bool = False, gender: str = 'male', 
        transform_action: str = 'both', target_type = "onehot") -> Dataset:
    """Load dataset class from files.

    Args:
        dataset_name (str, optional): The dataset name to use. Defaults to 'rsna-bone-age-kaggle'. [rsna-bone-age, rsna-bone-age-kaggle]
        defualt_path (str, optional): The default path to use. Defaults to ''.
        basedOnSex (bool, optional): If the dataset is based on the Gender or not. Defaults to False. 
        gender (str, optional): The gender of the dataset. Defaults to 'male'.
        transform_action (str, optional): Do transformation on witch dataset: 
            Options:
                None: No transformation for train and test.
                train: Transformation only for train.
                test: Transformation only for test.
                both: Trainsformation for train and test.
        target_type (str, optional): Data targeted. What type age data. Defaults to 'onehot'. 
                Options: ['onehot', 'minmax', 'zscore', 'real']

    Returns:
        datasets: The train and test datasets.
    """
    train_transform = data_augmentation() if transform_action in {'train', 'both'} else basic_transform()
    test_transform = data_augmentation() if transform_action in {'test', 'both'} else basic_transform()
    
    train_dataset = RSNATrainDataset(data_file = Path(defualt_path, f'dataset/{dataset_name}/boneage-training-dataset.csv'),
                           image_dir = Path(defualt_path, f'dataset/{dataset_name}/boneage-training-dataset/boneage-training-dataset/'),
                           basedOnSex = basedOnSex, gender = gender, transform = train_transform, target_type = target_type)


    test_dataset = RSNATestDataset(data_file = Path(defualt_path, f'dataset/{dataset_name}/boneage-test-dataset.csv'),
                           image_dir = Path(defualt_path, f'dataset/{dataset_name}/boneage-test-dataset/boneage-test-dataset/'),
                           basedOnSex = basedOnSex, gender = gender, transform=test_transform,
                           train_num_classes = train_dataset.num_classes, target_type = target_type)
    return train_dataset, test_dataset




# Test
def plot_data(img, r, c, i):
    plt.subplot(r, c, i)
    plt.imshow(img, cmap='gray')
    plt.axis('off')
    plt.tight_layout()


if __name__ == '__main__':

    train_dataset , test_dataset = data_handler(dataset_name = 'rsna-bone-age-kaggle', defualt_path = '', 
        basedOnSex = False, gender = 'male', target_type = 'minmax')

    batch_size, val_percent = 3, 0.2
    train_loader, val_loader, test_loader = data_wrapper(
                                                train_dataset = train_dataset, 
                                                test_dataset = test_dataset, 
                                                batch_size = batch_size, test_val_batch_size = 1,
                                                val_percent = val_percent, 
                                                shuffle = False, 
                                                num_workers = 1)

    print(len(train_loader.dataset))
    print(len(val_loader.dataset))
    print(len(test_loader.dataset))

    transf = torchvision.transforms.ToPILImage()
    count = 1
    row = 5
    col = 5
    with data_progress:
        
        for img_id, img, sex, target, boneage, ba_minmax, ba_zscore, boneage_onehot, num_classes in data_progress.track(train_loader):
            print(img_id, img.shape, sex, boneage.shape, ba_minmax.shape, ba_zscore)
            print(train_loader.dataset.dataset.reverse_min_max_normal(ba_minmax), train_loader.dataset.dataset.a_min, train_loader.dataset.dataset.a_max)
            print(target, train_loader.dataset.dataset.predict_compiler(target), train_loader.dataset.dataset.reverse_zscore_normal(ba_zscore))
            print(train_loader.dataset.dataset.train_data)

            # plot_data(transf(img[0]), row, col, count)
            # plot_data(img[0], row, col, count)
            # count += 1
            # if count == 25:
            #     break
            break

        # plt.tight_layout()
        # plt.savefig("C0555.png", dpi=300)
        # plt.show()

        print("---------------")
    
        for img_id, img, sex, target, boneage, ba_minmax, ba_zscore, boneage_onehot, _ in data_progress.track(test_loader):
            print(img_id, img.shape, sex, boneage, ba_minmax, ba_zscore)
            print(test_loader.dataset.reverse_min_max_normal(ba_minmax), test_loader.dataset.a_min, test_loader.dataset.a_max)
            print(target, test_loader.dataset.predict_compiler(target), test_loader.dataset.reverse_zscore_normal(ba_zscore))
            print(test_loader.dataset.test_data)
            
            break

