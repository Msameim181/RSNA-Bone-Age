
# System and utils for preprocessing
import logging
import sys
import os
from datetime import datetime
from pathlib import Path

# Deep learning libs
import torch

# Models
from models.MobileNet import MobileNet_V2
from models.ResNet import ResNet18, ResNet34, ResNet50, ResNet101, ResNet152
# Custom libs
from Train import trainer
from utils.dataloader import RSNATestDataset, RSNATrainDataset, data_wrapper
from utils.get_args import get_args
from Validation import validate
from Evaluation import evaluate
from utils.config_model import reload_model
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



# import wandb
# import glob

# wandb.save(glob.glob(f"tensorboard/*.pt.trace.json")[0], base_path=f"tensorboard")



# if __name__ == '__main__':

#     defualt_path = ''
#     train_dataset = RSNATrainDataset(data_file = Path(defualt_path, 'dataset/rsna-bone-age/boneage-training-dataset.csv'),
#                             image_dir = Path(defualt_path, 'dataset/rsna-bone-age/boneage-training-dataset/boneage-training-dataset/'),
#                             basedOnSex=False, gender='female')
#     # print(train_dataset.num_classes)
#     test_dataset = RSNATestDataset(data_file = Path(defualt_path, 'dataset/rsna-bone-age/boneage-test-dataset.csv'), 
#                             image_dir = Path(defualt_path, 'dataset/rsna-bone-age/boneage-test-dataset/boneage-test-dataset/'), 
#                             train_num_classes=train_dataset.num_classes, basedOnSex=False, gender='male')

#     _, _, test_loader = data_wrapper(
#                                                 train_dataset, 
#                                                 test_dataset, 
#                                                 1, 
#                                                 val_percent = 0.3, 
#                                                 shuffle = False, 
#                                                 num_workers = 1)
#     net = MobileNet_V2(pretrained = True, image_channels = 1, num_classes = train_dataset.num_classes).cuda()
#     reload_model(net, "./ResultModels/20220511_043706_MobileNetV2_Pre2/checkpoint_epoch18.pth")

#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     criterion = torch.nn.BCEWithLogitsLoss()
    
#     print(evaluate(net, test_loader, device, criterion))
    

from utils.rich_logger import make_console

if __name__ == '__main__':
    dataset_name = "rsna-bone-age" # rsna-bone-age-kaggle or rsna-bone-age
    basedOnSex = False
    gender = 'male'
    test_data = [
    {"jsonrpc": "2.0", "method": "sum", "params": [None, 1, 2, 4, False, True], "id": "1",},
    {"jsonrpc": "2.0", "method": "notify_hello", "params": [7]},
    {"jsonrpc": "2.0", "method": "subtract", "params": [42, 23], "id": "2"},
    ]
    console = make_console()
    console.print("Hello World (sdf) !asda 2%")
    console.log(test_data)
    console.print(f'\nDataSet: <{dataset_name}>\n'
                 f'Based On Gender: {basedOnSex}\n'
                 f'Targeted Gender: {gender}\n')
    console.out("DataSet", dataset_name)
    console.rule("[bold red]Chapter 2")

