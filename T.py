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



global_step = 4414
n_train = 8828
batch_size = 2
n = 2
es = 0
# n_train = n_train // batch_size
# # point = [0 if item == n else (n_train//n) * item  for item in range(1, n + 1)]
# co = 0
# print(point)
for i in range(n_train // batch_size):
    es += 1
    global_step += 1
    epoch_step = (global_step % (n_train // batch_size)) if global_step > (n_train // batch_size) else global_step
#     epoch_step = (global_step % n_train) if global_step >= n_train else global_step
#     if epoch_step in point:
#         print(global_step*batch_size)
#         co += 1
    print(epoch_step, es)
# print(co)