# Deep learning libs
import torch
from torchvision import models

from torchsummary import summary


class MobileNetV3(torch.nn.Module):
    def __init__(
        self, 
        image_channels: int, 
        num_classes: int = 100, 
        name: str = 'MobileNetV3', 
        pretrained: bool = True,
        input_size: int = 2,
        name_suffix: str = '',
    ) -> None:
        """MobileNetV3 class.

        Args:
            image_channels (int): Number of channels in the input image.
            num_classes (int, optional): Number of classes. Defaults to 100.
            name (str, optional): Name of the model. Defaults to 'MobileNetV2'.
            pretrained (bool, optional): Use pretrained model or not. Defaults to True.
            input_size (int, optional): Input size of the model. Defaults to 2.
        """

        super(MobileNetV3, self).__init__()

        self.type = '_Pre' if pretrained else ""
        self.name = name + self.type + name_suffix
        self.in_channels = image_channels
        self.num_classes = num_classes

        self.mobilenet_v2 = models.mobilenet_v3_large(pretrained=pretrained)

        self.mobilenet_v2.features[0][0] = torch.nn.Conv2d(image_channels, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)

        self.avgpool = torch.nn.AdaptiveAvgPool2d(1)

        in_features = self.mobilenet_v2.classifier[0].in_features
        self.add_feature = 0 if input_size <= 1 else input_size - 1

        self.mobilenet_v2.classifier = torch.nn.Sequential(

            # torch.nn.Dropout(0.2),
            torch.nn.Linear(in_features + 1, 2048),
            torch.nn.Hardswish(inplace=True),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(2048, 2048),
            torch.nn.Hardswish(inplace=True),

            torch.nn.Dropout(0.3),
            torch.nn.Linear(2048, 1024),
            torch.nn.Hardswish(inplace=True),

            torch.nn.Dropout(0.3),
            torch.nn.Linear(1024, 512),
            torch.nn.Hardswish(inplace=True),

            torch.nn.Linear(512, num_classes),

            # torch.nn.Dropout(0.5),
            # torch.nn.Linear(2048, num_classes),
            torch.nn.Sigmoid()
        )

    def forward(self, x) -> torch.Tensor:
        if self.add_feature > 0:
            y = x[1]
            x = x[0]
        
        x = self.mobilenet_v2.features(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        if self.add_feature > 0:
            x = torch.cat((x, y), dim=1)

        x = self.mobilenet_v2.classifier(x)

        return x


def MobileNet_V3(**kwargs) -> MobileNetV3:
    """
    Constructs a MobileNetV2 model
    Args:
        kwargs: Keyword arguments.
    """
    return MobileNetV3(**kwargs)




if __name__ == '__main__':
    model = MobileNet_V3(pretrained = True, image_channels = 1, num_classes = 229)
    # print(models.mobilenet_v3_large(pretrained=False))
    print(model.name)
    
    # model.cuda()
    # inp = torch.randn(1, 1, 500, 625).cuda()
    # sx = torch.randn(1, 1).cuda()
    # out = model([inp, sx])
    # print(out.shape)
    
