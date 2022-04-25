# System and utils for preprocessing
# import logging
# import os
# from pathlib import Path
# # Deep learning libs
# import numpy as np
# import pandas as pd
# import torch
# from PIL import Image
# # Custom libs
# from torch.utils.data import DataLoader, Dataset
# from utils.dataloader import RSNATestDataset, RSNATrainDataset
# from models.ResNet import ResNet18, ResNet50
# from models.MobileNet import MobileNetV2


# if __name__ == '__main__':
#     model = ResNet18(img_channel=1, num_classes=229)
#     print(model)
#     print(model.name)


# if __name__ == '__main__':
    
#     torch.cuda.empty_cache()
#     torch.cuda.memory_summary(device=None, abbreviated=False)
    
#     defualt_path = ''
#     train_dataset = RSNATrainDataset(data_file = Path(defualt_path, 'dataset/rsna-bone-age/boneage-training-dataset.csv'),
#                            image_dir = Path(defualt_path, 'dataset/rsna-bone-age/boneage-training-dataset/boneage-training-dataset/'),
#                            basedOnSex=True, gender='male')

    
#     num_classes = train_dataset.num_classes
#     net = ResNet18(image_channels=1, num_classes=num_classes)
#     device = 'cuda'
#     # net.to(device=device, dtype=torch.float32) 
#     # images = images.to(device=device, dtype=torch.float32)


#     learning_rate = 0.0001
#     epochs = 10
#     batch_size = 1
#     train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False, num_workers=1, pin_memory=True)
#     # test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=1, pin_memory=True)


#     for _, images, boneage, boneage_onehot, sex, num_classes in train_loader:
#         images = torch.unsqueeze(images, 1)

#         # net.cuda()
#         # images.cuda()
#         print(sex)
#         net.to(device=device, dtype=torch.float32) 
#         images = images.to(device=device, dtype=torch.float32)
#         sex = sex.to(device=device, dtype=torch.float32)
#         print(images.shape)
#         print(sex.shape)
#         out = net([images,sex])
#         print(out.shape)
#         print('----')
#         print(out)
#         print(boneage_onehot)
#         print('----')
#         print(boneage)
#         print(out.argmax(-1))
#         break
#     # train_net(net, device, train_loader, test_loader, 
#     #         epochs, batch_size, learning_rate)



# global_step = 4414
# n_train = 8828
# batch_size = 2
# n = 2
# es = 0
# # n_train = n_train // batch_size
# # # point = [0 if item == n else (n_train//n) * item  for item in range(1, n + 1)]
# # co = 0
# # print(point)
# for i in range(n_train // batch_size):
#     es += 1
#     global_step += 1
#     epoch_step = (global_step % (n_train // batch_size)) if global_step > (n_train // batch_size) else global_step
# #     epoch_step = (global_step % n_train) if global_step >= n_train else global_step
# #     if epoch_step in point:
# #         print(global_step*batch_size)
# #         co += 1
#     print(epoch_step, es)
# # print(co)


import logging
from pathlib import Path

import numpy as np
from tensorboardX import SummaryWriter
from models.MobileNet import MobileNet_V2
import torch

def tb_setup(config, log_dir:str = './tensorboard'):
    """
    Setup tensorboard logger
    """
    if not log_dir:
        log_dir = './tensorboard'
    net = config['net']
    model = config['model']
    name = config['name']
    device = config['device']
    epochs = config['epochs']
    batch_size = config['batch_size']
    learning_rate = config['learning_rate']
    save_checkpoint = config['save_checkpoint']
    amp = config['amp']

    # Create a run
    tb_logger = SummaryWriter(log_dir=Path(log_dir, name))

    tb_logger.add_text(tag='name', text_string=str(name), global_step=0)
    tb_logger.add_text(tag='model', text_string=str(model), global_step=0)
    tb_logger.add_text(tag='device', text_string=str(device), global_step=0)
    tb_logger.add_text(tag='epochs', text_string=str(epochs), global_step=0)
    tb_logger.add_text(tag='batch_size', text_string=str(batch_size), global_step=0)
    tb_logger.add_text(tag='learning_rate', text_string=str(learning_rate), global_step=0)
    tb_logger.add_text(tag='amp', text_string=str(amp), global_step=0)
    tb_logger.add_text(tag='save_checkpoint', text_string=str(save_checkpoint), global_step=0)

    tb_logger.add_graph(net.cuda(), ([torch.randn(batch_size, 1, 500, 625).cuda(), torch.randn(batch_size).cuda()], ))
    return tb_logger


def tb_log_training_step(tb_logger, loss, global_step, epoch, epoch_loss_step):
     # Logging
    tb_logger.add_scalar('Loss/Step Loss', loss.item(), global_step)
    tb_logger.add_scalar('Loss/Train Loss (Step)', epoch_loss_step, global_step)
    tb_logger.add_scalar('Process/Step', global_step, global_step)
    tb_logger.add_scalar('Process/Epoch', epoch, global_step)


def tb_log_training(tb_logger, epoch_loss, val_loss, epoch):
    # Logging
    tb_logger.add_scalar('Loss/Validation Loss (Epoch)', val_loss, epoch)
    tb_logger.add_scalar('Loss/Train Loss', epoch_loss, epoch)
    tb_logger.add_scalar('Loss/Epoch Loss', epoch_loss, epoch)
    logging.info(f'\nEpoch: {epoch} | Train Loss: {epoch_loss:.4f} | Validation Loss: {val_loss:.4f}\n')
    


def tb_log_validation(tb_logger, optimizer, val_loss, acc, 
    images, batch_size, global_step, epoch, net):
    # TensorBoard Storing the results
    tb_logger.add_scalar('Process/Learning Rate', optimizer.param_groups[0]['lr'], global_step)
    tb_logger.add_scalar('Loss/Validation Loss (Step)', val_loss, global_step)
    tb_logger.add_scalar('Accuracy/Validation Correct (Step)', acc, global_step)
    tb_logger.add_scalar('Accuracy/Correct %', acc * 100, global_step)
    tb_logger.add_scalar('Process/Step', global_step, global_step)
    tb_logger.add_scalar('Process/Epoch', epoch, global_step)
    # img_batch = images.cpu() if batch_size == 1 else [image.cpu() for image in images]
    # tb_logger.add_images('Data/Images', img_batch, global_step)

    for name, param in net.named_parameters():
        tb_logger.add_histogram(name, param.clone().cpu().data.numpy(), global_step)



if __name__ == '__main__':
    # Test
    tb_setup(dict(
            net = MobileNet_V2(pretrained = True, image_channels = 1, num_classes = 229), 
            epochs = 1000, 
            batch_size = 1, 
            learning_rate = 0.001,
            save_checkpoint = './checkpoints/', 
            amp = False,
            model = "net.name",
            name = "run_namessss",
            device = "devicessss",
            optimizer = "optimizer.__class__.__name__",
            criterion = "criterion.__class__.__name__"))
