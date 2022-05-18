# Deep learning libs
import torch
from torchvision import models

from torchsummary import summary

model_urls = {
    'mobilenet_v2': 'https://download.pytorch.org/models/mobilenet_v2-b0353104.pth',
}


class MobileNetV2(torch.nn.Module):
    def __init__(self, image_channels, num_classes = 100, name: str='MobileNetV2') -> None:
        super(MobileNetV2, self).__init__()

        self.name = name
        self.in_channels = image_channels
        self.num_classes = num_classes

        self.mobilenet_v2 = models.mobilenet_v2()

        self.mobilenet_v2.features[0][0] = torch.nn.Conv2d(image_channels, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        self.mobilenet_v2.classifier[1] = torch.nn.Linear(in_features=1281, out_features=num_classes, bias=True)


    def forward(self, x) -> torch.Tensor:
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


class MobileNetV2_Pre(torch.nn.Module):
    def __init__(self, image_channels, num_classes = 100, name: str='MobileNetV2_Pre') -> None:
        super(MobileNetV2_Pre, self).__init__()

        self.name = name
        self.in_channels = image_channels
        self.num_classes = num_classes

        self.mobilenet_v2 = models.mobilenet_v2(pretrained=True)

        self.mobilenet_v2.features[0][0] = torch.nn.Conv2d(image_channels, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)

        self.fc = torch.nn.Sequential(
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(1001, 512),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(512, num_classes),
        )


    def forward(self, x) -> torch.Tensor:
        y = x[1]
        x = x[0]
        
        x = self.mobilenet_v2(x)

        z = x
        y = torch.unsqueeze(y, 1).to(device='cuda', dtype=torch.float32)
        x = torch.cat((z, y), dim=1)

        x = self.fc(x)


        return x


class MobileNetV2_Pre2(torch.nn.Module):
    def __init__(self, image_channels, num_classes = 100, name: str='MobileNetV2_Pre2') -> None:
        super(MobileNetV2_Pre2, self).__init__()

        self.name = name
        self.in_channels = image_channels
        self.num_classes = num_classes

        self.mobilenet_v2 = models.mobilenet_v2(pretrained=True)

        self.mobilenet_v2.features[0][0] = torch.nn.Conv2d(image_channels, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)

        self.mobilenet_v2.classifier = torch.nn.Sequential(
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(1281, 512),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(512, num_classes),
        )


    def forward(self, x) -> torch.Tensor:
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


class MobileNetV2_Pre3(torch.nn.Module):
    def __init__(self, image_channels, num_classes = 100, name: str='MobileNetV2_Pre31') -> None:
        super(MobileNetV2_Pre3, self).__init__()

        self.name = name
        self.in_channels = image_channels
        self.num_classes = num_classes

        self.mobilenet_v2 = models.mobilenet_v2(pretrained=True)

        self.mobilenet_v2.features[0][0] = torch.nn.Conv2d(image_channels, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)

        self.mobilenet_v2.classifier = torch.nn.Sequential(
            torch.nn.ReLU(),
            # torch.nn.Dropout(0.2),
            torch.nn.Linear(1281, 512),
            torch.nn.ReLU(),
            # torch.nn.Dropout(0.5),
            torch.nn.Linear(512, num_classes),

            # torch.nn.Linear(256, 128),
            # torch.nn.Linear(128, 64),
            # torch.nn.Linear(64, 1),
        )


    def forward(self, x) -> torch.Tensor:
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



def MobileNet_V2(*, pretrained:bool = False, **kwargs) -> MobileNetV2:
    """
    Constructs a MobileNetV2 model
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    return MobileNetV2_Pre2(**kwargs) if pretrained else MobileNetV2(**kwargs)


if __name__ == '__main__':
    model = MobileNet_V2(pretrained = True, image_channels = 1, num_classes = 229)
    print(model)
    model.cuda()
    
    inp = torch.randn(1, 1, 500, 625).cuda()
    sx = torch.randn(1).cuda()
    # # print(inp.shape)
    # # print(sx.shape)
    # # # print(inp)
    # # # print(sx)
    out = model([inp, sx])
    # print(out.shape)
    print(model.name)


    # summary(model, (1, 500, 625,1 ), batch_size=1)