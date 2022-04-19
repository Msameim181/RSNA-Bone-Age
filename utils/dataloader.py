# System and utils for preprocessing
import logging
import os
from pathlib import Path

# Deep learning libs
import numpy as np
import pandas as pd
import torch
from PIL import Image
from rich.progress import (BarColumn, Progress, SpinnerColumn, TextColumn,
                           TimeElapsedColumn, TimeRemainingColumn)
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms

# Pre-initializing the loggers
progress = Progress(
    SpinnerColumn(finished_text="[bold blue]:heavy_check_mark:", style='blue'),
    TextColumn("[bold blue]{task.description}", justify="right"),
    BarColumn(bar_width=50, style='bar.back', complete_style='bar.complete', 
                finished_style='bar.finished', pulse_style='bar.pulse'),
    "[progress.percentage]{task.percentage:>3.1f}%",
    "•",
    TimeRemainingColumn(),
    "•",
    TimeElapsedColumn(),
    "•",
    "[progress.filesize]Passed: {task.completed} item",
    "•",
    "[progress.filesize.total]Total: {task.total:>.0f} item",
)

# wandb.login(key='0257777f14fecbf445207a8fdacdee681c72113a')


# Defining the dataset class
class RSNATrainDataset(Dataset):
    def __init__(self, data_file: str, image_dir: str, transform = None, 
                scale: float = 1.0, basedOnSex: bool=False, gender:str='male'):
        self.data_file = Path(data_file)
        self.image_dir = Path(image_dir)
        self.transform = transform
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'
        self.scale = scale
        self.basedOnSex = basedOnSex
        self.gender = gender

        # Proccessing data and csv file
        self.train_data = pd.read_csv(data_file)
        self.train_data['indx'] = range(len(self.train_data))

        # Dividing data based on gender
        if self.basedOnSex and self.gender == 'male':
            self.train_data_filtered = self.train_data[self.train_data['male'] == True]
        elif self.basedOnSex and self.gender == 'female':
            self.train_data_filtered = self.train_data[self.train_data['male'] == False]
        else:
            self.train_data_filtered = self.train_data

        # Number of classes for the one hot encoding
        self.num_classes = np.max(self.train_data['boneage']) + 1
        # One Hoting the bone age
        self.age_onehot  = torch.nn.functional.one_hot(torch.tensor(self.train_data['boneage']), num_classes = self.num_classes)

        if not os.path.exists(data_file):
            raise RuntimeError(f'No data file found in {data_file}.')
        if self.train_data.empty:
            raise RuntimeError('train data is empty file')

        logging.info(f'Creating dataset with {len(self.train_data)} samples')

    def __len__(self):
        return len(self.train_data_filtered)

    def __getitem__(self, index):
        img_id = self.train_data_filtered.iloc[index].id
        img_addr = Path(self.image_dir, f'{str(img_id)}.png')

        boneage = self.train_data_filtered.iloc[index].boneage

        onehot_index = self.train_data_filtered.iloc[index]['indx']
        boneage_onehot = self.age_onehot[onehot_index]

        sex = 1 if self.train_data.iloc[index].male else 0

        num_classes = self.num_classes

        assert os.path.exists(img_addr), f'Image {img_addr} does not exist'

        img = Image.open(img_addr)
        img = img.resize((500, 625))
        img = np.array(img)

        if self.transform is not None:
            augmentations = self.transform(image=img)
            img = augmentations["image"]

        # return img_id, img, boneage, boneage_onehot, sex
        return img_id, img, boneage, boneage_onehot, sex, num_classes


class RSNATestDataset(Dataset):
    def __init__(self, data_file: str, image_dir: str, transform = None, 
                scale: float = 1.0, basedOnSex: bool=False, gender:str='male'):
        self.data_file = Path(data_file)
        self.image_dir = Path(image_dir)
        self.transform = transform
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'
        self.scale = scale
        self.basedOnSex = basedOnSex
        self.gender = gender

        # Proccessing data and csv file
        self.test_data = pd.read_csv(self.data_file)

        # Dividing data based on gender
        if self.basedOnSex and self.gender == 'male':
            self.test_data_filtered = self.test_data[self.test_data['Sex'] == 'M']
        elif self.basedOnSex and self.gender == 'female':
            self.test_data_filtered = self.test_data[self.test_data['male'] == 'F']
        else:
            self.test_data_filtered = self.test_data

        if not os.path.exists(self.data_file):
            raise RuntimeError(f'No data file found in {data_file}.')
        if self.test_data.empty:
            raise RuntimeError('train data is empty file')

        logging.info(f'Creating dataset with {len(self.test_data)} samples')

    def __len__(self):
        return len(self.test_data_filtered)

    def __getitem__(self, index):

        img_id = self.test_data_filtered.iloc[index]['Case ID']
        img_addr = Path(self.image_dir, f'{str(img_id)}.png')

        sex = 1 if self.test_data_filtered.iloc[index]['Sex'] == 'M' else 0

        assert os.path.exists(img_addr), f'Image {img_addr} does not exist'

        img = Image.open(img_addr)
        img = img.resize((500, 625))
        img = np.array(img)

        if self.transform is not None:
            augmentations = self.transform(image=img)
            img = augmentations["image"]

        return img_id, img, sex


# Data Packaging
def data_wrapper(train_dataset, test_dataset, batch_size: int, val_percent: float = 0.2, shuffle: bool = True, num_workers: int = 1):
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
    val_loader = DataLoader(dataset = val_set, batch_size = 1, shuffle = shuffle, num_workers = num_workers, pin_memory = True)

    test_loader = DataLoader(dataset = test_dataset, batch_size = batch_size, shuffle = shuffle, num_workers = num_workers, pin_memory = True)

    return train_loader, val_loader, test_loader



if __name__ == '__main__':
    defualt_path = ''
    train_dataset = RSNATrainDataset(data_file = Path(defualt_path, 'dataset/rsna-bone-age/boneage-training-dataset.csv'),
                            image_dir = Path(defualt_path, 'dataset/rsna-bone-age/boneage-training-dataset/boneage-training-dataset/'),
                            basedOnSex=True, gender='female')

    test_dataset = RSNATestDataset(data_file = defualt_path + 'dataset/rsna-bone-age/boneage-test-dataset.csv',
                            image_dir = defualt_path + 'dataset/rsna-bone-age/boneage-test-dataset/boneage-test-dataset/',
                            basedOnSex=True, gender='male')

    train_loader = DataLoader(dataset=train_dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, num_workers=8, pin_memory=True)





    # with progress:
    #     for img_id, img, boneage, sex, num_classes in progress.track(train_loader):
    #         # print(torch.argmax(boneage_onehot), boneage, boneage_onehot.shape)
    #         # images = torch.unsqueeze(img, 1)
    #         print(int(num_classes))
    #         break
    #         ...


    # with progress:
    #     for img_id, img, sex in progress.track(test_loader):
    #         # print(img_id, img, sex)
    #         ...

