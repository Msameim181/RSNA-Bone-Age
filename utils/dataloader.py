# System and utils for preprocessing
import logging
import os
from pathlib import Path
import matplotlib.pyplot as plt
# Deep learning libs
import numpy as np
import pandas as pd
import torch
from PIL import Image
from rich.console import Console
from rich.progress import (BarColumn, Progress, SpinnerColumn, TextColumn,
                           TimeElapsedColumn, TimeRemainingColumn)
from torch.utils.data import DataLoader, Dataset, random_split
import torchvision
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

# Pre-initializing the loggers
console = Console()
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

        a_min, a_max = 1, 228
        ba_norm = self.train_data['boneage'].copy()
        ba_norm -= a_min
        ba_norm /= a_max
        self.train_data['ba_norm'] = ba_norm

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

        console.print(f'[INFO]: Creating dataset with {len(self.train_data)} samples')

    def __len__(self):
        return len(self.train_data_filtered)

    def __getitem__(self, index):
        img_id = self.train_data_filtered.iloc[index].id
        img_addr = Path(self.image_dir, f'{str(img_id)}.png')

        boneage = self.train_data_filtered.iloc[index].boneage
        ba_norm = self.train_data_filtered.iloc[index].ba_norm

        onehot_index = self.train_data_filtered.iloc[index]['indx']
        boneage_onehot = self.age_onehot[onehot_index]

        sex = 1 if self.train_data_filtered.iloc[index].male else 0

        num_classes = self.num_classes

        assert os.path.exists(img_addr), f'Image {img_addr} does not exist'

        img = Image.open(img_addr)
        # img = img.resize((500, 625))
        img = np.array(img)
        
        if self.transform is not None:
            augmentations = self.transform(image=img)
            img = augmentations["image"]

        # return img_id, img, boneage, boneage_onehot, sex
        return img_id, img, boneage, boneage_onehot, ba_norm, sex, num_classes


class RSNATestDataset(Dataset):
    def __init__(self, data_file: str, image_dir: str, train_num_classes: int, transform = None,
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
        self.test_data['indx'] = range(len(self.test_data))

        # Dividing data based on gender
        if self.basedOnSex and self.gender == 'male':
            self.test_data_filtered = self.test_data[self.test_data['male'] == True]
        elif self.basedOnSex and self.gender == 'female':
            self.test_data_filtered = self.test_data[self.test_data['male'] == False]
        else:
            self.test_data_filtered = self.test_data

        if 'boneage' in self.test_data.keys():
            a_min, a_max = 1, 228
            ba_norm = self.test_data_filtered['boneage'].copy()
            ba_norm -= a_min
            ba_norm /= a_max
            self.test_data_filtered['ba_norm'] = ba_norm
            # One Hoting the bone age
            self.age_onehot  = torch.nn.functional.one_hot(torch.tensor(self.test_data['boneage']), num_classes = train_num_classes)


        if not os.path.exists(self.data_file):
            raise RuntimeError(f'No data file found in {data_file}.')
        if self.test_data.empty:
            raise RuntimeError('train data is empty file')

        console.print(f'[INFO]: Creating dataset with {len(self.test_data)} samples')

    def __len__(self):
        return len(self.test_data_filtered)

    def __getitem__(self, index):

        img_id = self.test_data_filtered.iloc[index]['Case ID']
        img_addr = Path(self.image_dir, f'{str(img_id)}.png')

        sex = 1 if self.test_data_filtered.iloc[index].male else 0

        boneage = 0
        ba_norm = 0
        boneage_onehot = 0
        if 'boneage' in self.test_data_filtered.keys():
            boneage = self.test_data_filtered.iloc[index].boneage
            ba_norm = self.test_data_filtered.iloc[index].ba_norm

            onehot_index = self.test_data_filtered.iloc[index]['indx']
            boneage_onehot = self.age_onehot[onehot_index]

        assert os.path.exists(img_addr), f'Image {img_addr} does not exist'

        img = Image.open(img_addr)
        # img = img.resize((500, 625))
        img = np.array(img)

        if self.transform is not None:
            augmentations = self.transform(image=img)
            img = augmentations["image"]

        return img_id, img, boneage, boneage_onehot, ba_norm, sex


def data_augmentation():
    return A.Compose([
        A.Resize(625, 500),
        A.ColorJitter(brightness=0.0, contrast=0.5, saturation=0.5, hue=0.5, p=0.9, always_apply=True),
        A.Rotate(limit=45, p=0.5),
        ToTensorV2(),
    ])

# Data Packaging
def data_wrapper(train_dataset: Dataset, test_dataset: Dataset, 
        batch_size: int, test_val_batch_size: int = 1, val_percent: float = 0.2, 
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
        basedOnSex: bool = False, gender: str = 'male') -> Dataset:
    """Load dataset class from files.

    Args:
        dataset_name (str, optional): The dataset name to use. Defaults to 'rsna-bone-age-kaggle'. [rsna-bone-age, rsna-bone-age-kaggle]
        defualt_path (str, optional): The default path to use. Defaults to ''.
        basedOnSex (bool, optional): If the dataset is based on the Gender or not. Defaults to False. 
        gender (str, optional): The gender of the dataset. Defaults to 'male'.

    Returns:
        datasets: The train and test datasets.
    """
    train_dataset = RSNATrainDataset(data_file = Path(defualt_path, f'dataset/{dataset_name}/boneage-training-dataset.csv'),
                           image_dir = Path(defualt_path, f'dataset/{dataset_name}/boneage-training-dataset/boneage-training-dataset/'),
                           basedOnSex = basedOnSex, gender = gender, transform=data_augmentation())

    test_dataset = RSNATestDataset(data_file = Path(defualt_path, f'dataset/{dataset_name}/boneage-test-dataset.csv'),
                           image_dir = Path(defualt_path, f'dataset/{dataset_name}/boneage-test-dataset/boneage-test-dataset/'),
                           basedOnSex = basedOnSex, gender = gender, transform=data_augmentation(),
                           train_num_classes = train_dataset.num_classes)
    return train_dataset, test_dataset




# ----------------------------------------------------------------- #
# |                                                               | #
# |                          Testing                              | #
# |                                                               | #
# ----------------------------------------------------------------- #

def plot_data(img, r, c, i):
    plt.subplot(r, c, i)
    plt.imshow(img, cmap='gray')
    plt.axis('off')
    plt.tight_layout()


if __name__ == '__main__':
    defualt_path = ''
    # train_dataset = RSNATrainDataset(data_file = Path(defualt_path, 'dataset/rsna-bone-age-kaggle/boneage-training-dataset.csv'),
    #                         image_dir = Path(defualt_path, 'dataset/rsna-bone-age-kaggle/boneage-training-dataset/boneage-training-dataset/'),
    #                         basedOnSex=False, gender='female', transform=data_augmentation())
    # # print(train_dataset.num_classes)
    # test_dataset = RSNATestDataset(data_file = Path(defualt_path, 'dataset/rsna-bone-age-kaggle/boneage-test-dataset.csv'), 
    #                         image_dir = Path(defualt_path, 'dataset/rsna-bone-age-kaggle/boneage-test-dataset/boneage-test-dataset/'), 
    #                         train_num_classes=train_dataset.num_classes, basedOnSex=False, gender='male', 
    #                         transform=data_augmentation())

    train_dataset , test_dataset = data_handler(dataset_name = 'rsna-bone-age-kaggle', defualt_path = '', 
        basedOnSex = False, gender = 'male')

    # train_loader = DataLoader(dataset=train_dataset, batch_size=5, shuffle=False, num_workers=1, pin_memory=True)
    # test_loader = DataLoader(dataset=test_dataset, batch_size=6, shuffle=False, num_workers=1, pin_memory=True)
    batch_size, val_percent = 1, 0.2
    train_loader, val_loader, test_loader = data_wrapper(
                                                train_dataset = train_dataset, 
                                                test_dataset = test_dataset, 
                                                batch_size = batch_size, test_val_batch_size = 1,
                                                val_percent = val_percent, 
                                                shuffle = False, 
                                                num_workers = 1)

    transf = torchvision.transforms.ToPILImage()
    # show_dataset(train_dataset)
    # count = 1
    # row = 5
    # col = 5
    # with progress:
    #     for img_id, img, boneage, boneage_onehot, ba_norm, sex, num_classes in progress.track(train_loader):
    #         # print(torch.argmax(boneage_onehot), boneage, boneage_onehot.shape)
    #         # images = torch.unsqueeze(img, 1)
    #         # print(img.shape)
    #         # break
    #         a_min, a_max = 1, 228
    #         t = ba_norm * a_max
    #         t += a_min
    #         # print(boneage, ba_norm, t)
    #         plot_data(transf(img[0]), row, col, count)
    #         # plot_data(img[0], row, col, count)
    #         count += 1
    #         if count == 25:
    #             break
    #         ...
    # plt.tight_layout()
    # plt.savefig("C0555.png", dpi=300)
    # plt.show()
    print(len(train_loader.dataset))
    print(len(val_loader.dataset))
    print(len(test_loader.dataset))

    with progress:
        for img_id, img, boneage, boneage_onehot, ba_norm, sex in progress.track(test_loader):
            print(img_id, img.shape, sex, boneage, ba_norm)
            break
        

        for img_id, img, boneage, boneage_onehot, ba_norm, sex, num_classes in progress.track(train_loader):
            print(img_id, img.shape, sex, boneage, ba_norm)
            break

