
# System and utils for preprocessing
import logging
import sys
import os
from datetime import datetime
from pathlib import Path
import pandas as pd
# Deep learning libs
# import torch

# Models
# from models.MobileNet import MobileNet_V2
# from models.ResNet import ResNet18, ResNet34, ResNet50, ResNet101, ResNet152
# Custom libs
# from Train import trainer
from utils.dataloader import *
# from utils.get_args import get_args
# from Validation import validate
# from Evaluation import evaluate
# from utils.config_model import *
# from torch.utils.data import DataLoader, Dataset, random_split
# if __name__ == '__main__':
#     model = ResNet18(img_channel=1, num_classes=229)
#     print(model)
#     print(model.name)


# if __name__ == '__main__':
    
#     torch.cuda.empty_cache()
#     torch.cuda.memory_summary(device=None, abbreviated=False)
    
#     defualt_path = ''
#     train_dataset = RSNATrainDataset(data_file = Path(defualt_path, 'dataset/rsna-bone-age-kaggle/boneage-training-dataset.csv'),
#                             image_dir = Path(defualt_path, 'dataset/rsna-bone-age-kaggle/boneage-training-dataset/boneage-training-dataset/'),
#                             basedOnSex=False, gender='female')

    
#     num_classes = train_dataset.num_classes
#     net = ResNet50(image_channels=1, num_classes=num_classes)
#     device = 'cuda'
#     # net.to(device=device, dtype=torch.float32) 
#     # images = images.to(device=device, dtype=torch.float32)


#     learning_rate = 0.0001
#     epochs = 10
#     batch_size = 8
#     train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False, num_workers=1, pin_memory=True)
#     # test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=1, pin_memory=True)


#     for _, images, boneage, boneage_onehot, sex, num_classes in train_loader:
#         images = torch.unsqueeze(images, 1)
#         sex = torch.unsqueeze(sex, 1)

#         # net.cuda()
#         # images.cuda()
#         print(sex.shape)
#         net.to(device=device, dtype=torch.float32) 
#         images = images.to(device=device, dtype=torch.float32)
#         sex = sex.to(device=device, dtype=torch.float32)
#         print(images.shape)
#         print(sex.shape)
#         out = net([images,sex])
#         print(out.shape)
#         print('----')
#         print(out)
#         print(boneage_onehot.shape)
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
#     val_point = [n_train_batch if item == val_repeat else ((n_train_batch//val_repeat) * item)  for item in range(1, val_repeat + 1)]
    
#     # n_train_batch += 1
#     # epoch_step = (global_step % n_train_batch) if global_step >= n_train_batch else global_step
#     if epoch_step in val_point:
#         print(val_point)
#         print('val_point(epoch_step):', epoch_step)
#         print('val_point(global_step): ', global_step)
#         return True
# if __name__ == '__main__':
#     # Test



#     val_repeat = 3
#     batch_size = 1
#     n_train = 8828
#     global_step = 0


#     print(f"n_train // batch_size: {n_train // batch_size}")
#     print(f"n_train % batch_size: {n_train % batch_size}")
#     print(packaging(batch_size=batch_size, n_train=n_train))
#     # print(8828 % 2)
#     for epoch in range(2):
#         print(f"epoch: {epoch}")
#         epoch_step = 0
#         for _ in range(packaging(batch_size, n_train)):
#             global_step += 1
#             epoch_step += 1
#             if ch := vel_section(batch_size, n_train, val_repeat, global_step):
#                 log_epoch_loss = ((((epoch_step - 1) * batch_size) + (n_train % batch_size))) if n_train // batch_size == (epoch_step - 1) else ((epoch_step * batch_size))
#                 print(f"val_loss T {epoch_step} / {epoch_step * batch_size} / {global_step} / {log_epoch_loss}")
#             # print(i)



# import wandb
# import glob

# wandb.save(glob.glob(f"tensorboard/*.pt.trace.json")[0], base_path=f"tensorboard")



# if __name__ == '__main__':

    # defualt_path = ''
    # train_dataset = RSNATrainDataset(data_file = Path(defualt_path, 'dataset/rsna-bone-age/boneage-training-dataset.csv'),
    #                         image_dir = Path(defualt_path, 'dataset/rsna-bone-age/boneage-training-dataset/boneage-training-dataset/'),
    #                         basedOnSex=False, gender='female')
    # # print(train_dataset.num_classes)
    # test_dataset = RSNATestDataset(data_file = Path(defualt_path, 'dataset/rsna-bone-age/boneage-test-dataset.csv'), 
    #                         image_dir = Path(defualt_path, 'dataset/rsna-bone-age/boneage-test-dataset/boneage-test-dataset/'), 
    #                         train_num_classes=train_dataset.num_classes, basedOnSex=False, gender='male')

    # dataset_name = "rsna-bone-age" # rsna-bone-age-kaggle or rsna-bone-age
    # basedOnSex = False
    # gender='male'

    # train_dataset , test_dataset = data_handler(dataset_name = dataset_name, defualt_path = '', 
    #                                     basedOnSex = basedOnSex, gender = gender, transform_action = 'train')
    # num_classes = train_dataset.num_classes 

    # train_loader, _, test_loader = data_wrapper(
    #                                             train_dataset, 
    #                                             test_dataset, 
    #                                             1, 
    #                                             val_percent = 0.3, 
    #                                             shuffle = False, 
    #                                             num_workers = 1)
    # net = MobileNet_V3(pretrained = True, image_channels = 1, num_classes = train_dataset.num_classes).cuda()
    # reload_model(net, "./ResultModels/20220523_110557_MobileNetV3_Pre/checkpoint_epoch25.pth")

    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # criterion = torch.nn.BCEWithLogitsLoss()
    
    # print(evaluate(net, test_loader, device, criterion))


    # from pathlib import Path
    # import pandas as pd
    # import numpy as np
    # dataset_name = "rsna-bone-age-neu"
    # pathss = Path("dataset/rsna-bone-age-neu/boneage-training-dataset.csv")
    # image_dir = Path("dataset/rsna-bone-age-neu/boneage-training-dataset/boneage-training-dataset/")
    # train_data = pd.read_csv(pathss)
    # radiographs_images = np.array([f.stem for f in image_dir.glob('*.png')]).astype('int64')
    # print(train_data)
    # train_data = train_data[train_data.id.isin(radiographs_images)]
    # train_data.to_csv("dataset/rsna-bone-age-neu/boneage-training-dataset2.csv", index=False)
import argparse
from utils.config_model import load_model

def make_fake_args():
    args = argparse.ArgumentParser()
    return args.parse_args()

import cv2
def visualize_attn(I, c):
    # Image
    img = I.permute((1,2,0)).cpu().numpy()
    # Heatmap
    B, C, H, W = c.size()
    a = torch.nn.functional.softmax(c.view(B,C,-1), dim=2).view(B,C,H,W)
    up_factor = 40/H
    print(up_factor, I.size(), c.size())
    if up_factor > 1:
        a = torch.nn.functional.interpolate(a, scale_factor=up_factor, mode='bilinear', align_corners=False)
    attn = utils.make_grid(a, nrow=4, normalize=True, scale_each=True)
    attn = attn.permute((1,2,0)).mul(255).byte().cpu().numpy()
    attn = cv2.applyColorMap(attn, cv2.COLORMAP_JET)
    attn = cv2.cvtColor(attn, cv2.COLOR_BGR2RGB)
    # Resize attn to match the image
    attn = cv2.resize(attn, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_CUBIC)

    # Add the heatmap to the image
    # vis = 0.4 * attn
    vis = 0.8 * img + 0.2 * attn
    return torch.from_numpy(vis).permute(2,0,1)


if __name__=='__main__':
    from utils.config_model import *
    from utils.dataloader import *
    dataset_name = "rsna-bone-age-neu" # rsna-bone-age-kaggle or rsna-bone-age
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
    
    model = load_model("./ResultModels/20220629_151413_MobileNetV3_Attention_Pre_MSE_G-32_Atten/checkpoint_model.pth").cuda()
    # model = load_model("./ResultModels/20220626_100441_MobileNetV3_Attention_Pre_MSE_G-32_Atten/checkpoint_model.pth").cuda()
    # reload_model(model, "./ResultModels/20220619_172133_MobileNetV3_Pre_MSE_G-FC32_RSNA/checkpoint_epoch17.pth")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # print(model)

    import matplotlib.pyplot as plt
    from torchvision import models, utils
    from tqdm import tqdm
    model.eval()
    n_eval = len(test_loader.dataset)
    co = 0
    for idx, images, gender, target, boneage, ba_minmax, ba_zscore, boneage_onehot, _ in tqdm(test_loader, total = n_eval, desc='Evaluation Round...', unit = 'img', leave=False):

        images = images.to(device = device, dtype = torch.float32)

        gender = torch.unsqueeze(gender, 1)
        gender = gender.to(device = device, dtype = torch.float32)

        target = target.to(device = device, dtype = torch.float32)
        boneage = boneage.to(device = device, dtype = torch.float32)
        ba_minmax = ba_minmax.to(device = device, dtype = torch.float32)


        with torch.no_grad():
            age_pred, c1, c2, c3 = model([images, gender])
            pred = test_loader.dataset.predict_compiler(age_pred)
            print(f"Pred: {pred.cpu().numpy().item()}")
            print(f"True: {boneage.cpu().numpy().item()}")
            
            I = utils.make_grid(images, nrow=4, normalize=True, scale_each=True)

            print(I.shape, c1.shape, c2.shape, c3.shape)
            attn1 = visualize_attn(I, c1)
            attn2 = visualize_attn(I, c2)
            attn3 = visualize_attn(I, c3)
        
            # plot image and attention maps
            plt.figure(figsize=(10, 5))
            plt.subplot(2, 4, 1)
            plt.imshow(I.permute((1,2,0)).cpu().numpy())
            plt.subplot(2, 4, 2)
            plt.imshow(utils.make_grid(c1, nrow=4).permute(1, 2, 0).cpu().numpy(), cmap='gray', )
            plt.colorbar()
            plt.subplot(2, 4, 3)
            plt.imshow(utils.make_grid(c2, nrow=4).permute(1, 2, 0).cpu().numpy(), cmap='gray', )
            plt.colorbar()
            plt.subplot(2, 4, 4)
            plt.imshow(utils.make_grid(c3, nrow=4).permute(1, 2, 0).cpu().numpy(), cmap='gray', )
            plt.colorbar()
            plt.subplot(2, 4, 5)
            plt.imshow(attn1.permute((1,2,0)).cpu().numpy(), cmap=plt.cm.get_cmap('jet'))
            plt.colorbar(extend='both')
            plt.subplot(2, 4, 6)
            plt.imshow(attn2.permute((1,2,0)).cpu().numpy(), cmap=plt.cm.get_cmap('jet'))
            plt.colorbar(extend='both')
            plt.subplot(2, 4, 7)
            plt.imshow(attn3.permute((1,2,0)).cpu().numpy(), cmap=plt.cm.get_cmap('jet'))
            plt.colorbar(extend='both')
            plt.show()

            if co == 5:
                break
            co += 1
            # break