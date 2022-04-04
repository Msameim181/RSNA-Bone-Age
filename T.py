# System and utils for preprocessing
import logging
import os
from pathlib import Path
# Deep learning libs
import numpy as np
import pandas as pd
import torch
from PIL import Image
# Custom libs
from torch.utils.data import DataLoader, Dataset
from utils.dataloader import RSNATestDataset, RSNATrainDataset
from ResNet.resnet_model import ResNet, Block

def ResNet50(img_channel=3, num_classes=1000):
    return ResNet(Block, [3, 4, 6, 3], img_channel, num_classes)

if __name__ == '__main__':
    
    torch.cuda.empty_cache()
    torch.cuda.memory_summary(device=None, abbreviated=False)
    
    defualt_path = ''
    train_dataset = RSNATrainDataset(data_file = Path(defualt_path, 'dataset/rsna-bone-age/boneage-training-dataset.csv'),
                           image_dir = Path(defualt_path, 'dataset/rsna-bone-age/boneage-training-dataset/boneage-training-dataset/'),
                           basedOnSex=True, gender='male')

    
    num_classes = train_dataset.num_classes
    net = ResNet50(img_channel=1, num_classes=num_classes)
    device = 'cuda'
    # net.to(device=device, dtype=torch.float32) 
    # images = images.to(device=device, dtype=torch.float32)


    learning_rate = 0.0001
    epochs = 10
    batch_size = 1
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False, num_workers=1, pin_memory=True)
    # test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=1, pin_memory=True)


    for _, images, boneage, boneage_onehot, sex, num_classes in train_loader:
        images = torch.unsqueeze(images, 1)

        # net.cuda()
        # images.cuda()
        print(sex)
        net.to(device=device, dtype=torch.float32) 
        images = images.to(device=device, dtype=torch.float32)
        sex = sex.to(device=device, dtype=torch.float32)
        print(images.shape)
        print(sex.shape)
        out = net([images,sex])
        print(out.shape)
        print('----')
        print(out)
        print(boneage_onehot)
        print('----')
        print(boneage)
        print(out.argmax(-1))
        break
    # train_net(net, device, train_loader, test_loader, 
    #         epochs, batch_size, learning_rate)