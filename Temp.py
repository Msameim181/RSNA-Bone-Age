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



# def packaging(batch_size, n_train):
#     batch_num = n_train // batch_size
#     if n_train % batch_size != 0:
#         batch_num += 1
#     return batch_num

# def vel_section(batch_size, n_train, val_repeat, global_step):
#     n_train_batch = n_train // batch_size
#     val_point = [0 if item == val_repeat else ((n_train_batch//val_repeat) * item)  for item in range(1, val_repeat + 1)]
#     n_train_batch += 1
#     epoch_step = (global_step % n_train_batch) if global_step >= n_train_batch else global_step
#     if epoch_step in val_point:
#         print('val_point(epoch_step):', epoch_step)
#         print('val_point(global_step): ', global_step)
#         return True
# if __name__ == '__main__':
#     # Test



#     val_repeat = 2
#     batch_size = 6
#     n_train = 8828
#     global_step = 0


#     print(f"n_train // batch_size: {n_train // batch_size}")
#     print(f"n_train % batch_size: {n_train % batch_size}")
#     print(packaging(batch_size=batch_size, n_train=n_train))
#     # print(8828 % 2)
#     for epoch in range(1):
#         print(f"epoch: {epoch}")
#         epoch_step = 0
#         for i in range(packaging(batch_size, n_train)):
#             global_step += 1
#             epoch_step += 1
#             if ch := vel_section(batch_size, n_train, val_repeat, global_step):
#                 log_epoch_loss = ((((epoch_step - 1) * batch_size) + (n_train % batch_size))) if n_train // batch_size == (epoch_step - 1) else ((epoch_step * batch_size))
#                 print(f"val_loss T {epoch_step} / {epoch_step * batch_size} / {global_step} / {log_epoch_loss}")
#             # print(i)



import wandb
import glob

wandb.save(glob.glob(f"tensorboard/*.pt.trace.json")[0], base_path=f"tensorboard")