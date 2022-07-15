import argparse

import cv2
import matplotlib.pyplot as plt
from torchvision import models, utils
from tqdm import tqdm

from utils.config_model import *
from utils.dataloader import *


def make_fake_args():
    args = argparse.ArgumentParser()
    return args.parse_args()


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
    vis = 0.4 * img + 0.6 * attn
    return torch.from_numpy(vis).permute(2,0,1)


if __name__=='__main__':

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
