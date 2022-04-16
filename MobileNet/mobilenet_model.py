# Deep learning libs
import torch
from torchvision import models

from torchsummary import summary

model_urls = {
    'mobilenet_v2': 'https://download.pytorch.org/models/mobilenet_v2-b0353104.pth',
}


class MobileNetV2(torch.nn.Module):
    def __init__(self, img_channel, num_classes = 100, name: str='MobileNetV2') -> None:
        super(MobileNetV2, self).__init__()

        self.name = name
        self.n_channels = img_channel
        self.num_classes = num_classes

        self.mobilenet_v2 = models.mobilenet_v2()

        self.mobilenet_v2.features[0][0] = torch.nn.Conv2d(img_channel, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        self.mobilenet_v2.classifier[1] = torch.nn.Linear(in_features=1281, out_features=num_classes, bias=True)


    def forward(self, x):
        y = x[1]
        x = x[0]
        
        x = self.mobilenet_v2.features(x)

        x = torch.nn.functional.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)

        z = x
        y = torch.unsqueeze(y, 1).to(device='cuda', dtype=torch.float32)
        x = torch.cat((z, y), dim=1)

        x = self.mobilenet_v2.classifier(x)


        return x


if __name__ == '__main__':
    model = MobileNetV2(img_channel=1, num_classes=229)
    # print(model)
    model.cuda()
    
    inp = torch.randn(5, 1, 500, 625).cuda()
    sx = torch.randn(5).cuda()
    # print(inp.shape)
    # print(sx.shape)
    # # print(inp)
    # # print(sx)
    out = model([inp, sx])
    # print(out.shape)


    # summary(model, (1, 500, 625,1 ), batch_size=1)