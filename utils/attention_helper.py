
import argparse
from config_model import load_model
import torch



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

