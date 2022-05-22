# Deep learning libs
import torch
import torchvision.models as models



class ResNet(torch.nn.Module):
    def __init__(
        self,
        image_channels: int = 3,
        num_classes: int = 1000,
        name: str='ResNet',
        pretrained: bool = True,
        input_size: int = 2,
        name_suffix: str = '',
    ) -> None:
        super(ResNet, self).__init__()

        self.type = '_Pre' if pretrained else ""
        self.name = name + self.type + name_suffix
        self.in_channels = image_channels
        self.num_classes = num_classes
        self.inplanes = 64

        if name == "ResNet18":
            self.resnet = models.resnet18(pretrained=pretrained)
        elif name == "ResNet34":
            self.resnet = models.resnet34(pretrained=pretrained)
        elif name == "ResNet50":
            self.resnet = models.resnet50(pretrained=pretrained)
        elif name == "ResNet101":
            self.resnet = models.resnet101(pretrained=pretrained)
        elif name == "ResNet152":
            self.resnet = models.resnet152(pretrained=pretrained)

        self.resnet.conv1 = torch.nn.Conv2d(self.in_channels, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)

        in_features = self.resnet.fc.in_features
        self.add_feature = 0 if input_size <= 1 else input_size - 1
        self.resnet.fc = torch.nn.Sequential(
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(in_features + self.add_feature, 512),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(512, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.add_feature > 0:
            y = x[1]
            x = x[0]

        # See note [TorchScript super()]
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)
        x = self.resnet.avgpool(x)
        x = torch.flatten(x, 1)

        if self.add_feature > 0:
            y = torch.unsqueeze(y, 1).to(device='cuda', dtype=torch.float32)
            x = torch.cat((x, y), dim=1)

        x = self.resnet.fc(x)


        return x


def ResNet18(pretrained = False, image_channels = 3, num_classes = 1000, **kwargs) -> ResNet:
    
    return ResNet(pretrained = pretrained, image_channels = image_channels, 
                num_classes = num_classes, name='ResNet18', **kwargs)

def ResNet34(pretrained = False, image_channels = 3, num_classes = 1000, **kwargs) -> ResNet:
    
    return ResNet(pretrained = pretrained, image_channels = image_channels, 
                num_classes = num_classes, name='ResNet34', **kwargs)

def ResNet50(pretrained = False, image_channels = 3, num_classes = 1000, **kwargs) -> ResNet:
    
    return ResNet(pretrained = pretrained, image_channels = image_channels, 
                num_classes = num_classes, name='ResNet50', **kwargs)

def ResNet101(pretrained = False, image_channels = 3, num_classes = 1000, **kwargs) -> ResNet:
    
    return ResNet(pretrained = pretrained, image_channels = image_channels, 
                num_classes = num_classes, name='ResNet101', **kwargs)

def ResNet152(pretrained = False, image_channels = 3, num_classes = 1000, **kwargs) -> ResNet:
    
    return ResNet(pretrained = pretrained, image_channels = image_channels, 
                num_classes = num_classes, name='ResNet152', **kwargs)



if __name__ == '__main__':
    model = ResNet18(pretrained = True, image_channels = 1, num_classes = 229, input_size = 2)
    # print(model)
    print(model.name)

    
    # model.cuda()
    # inp = torch.randn(1, 1, 500, 625).cuda()
    # sx = torch.randn(1).cuda()
    # out = model([inp, sx])
    # print(out.shape)