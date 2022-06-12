import torch
from torchvision import models



class VGGNet(torch.nn.Module):
    def __init__(
        self, 
        image_channels: int = 3, 
        num_classes: int = 100, 
        name: str='VGGNet',
        pretrained: bool = True,
        input_size: int = 2,
        name_suffix: str = '',
        **kwargs
    ) -> None:
        super(VGGNet, self).__init__()

        self.type = '_Pre' if pretrained else ""
        self.name = name + self.type + name_suffix
        self.in_channels = image_channels
        self.num_classes = num_classes
        self.add_feature = 0 if input_size <= 1 else input_size - 1


        if name == "VGGNet11":
            self.vggnet = models.vgg11(pretrained=pretrained)
        elif name == "VGGNet13":
            self.vggnet = models.vgg13(pretrained=pretrained)
        elif name == "VGGNet16":
            self.vggnet = models.vgg16(pretrained=pretrained)
        elif name == "VGGNet19":
            self.vggnet = models.vgg19(pretrained=pretrained)

        self.vggnet.features[0] = torch.nn.Conv2d(image_channels, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

        self.vggnet.fc1 = torch.nn.Sequential(
            torch.nn.Linear(1, 16),
            torch.nn.ReLU(),
        )

        self.vggnet.classifier = torch.nn.Sequential(
            torch.nn.Linear((512 * 7 * 7) + 1, 4096),
            torch.nn.ReLU(True),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(4096, 2048),
            torch.nn.ReLU(True),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(2048, 1024),
            torch.nn.ReLU(True),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(1024, 512),
            torch.nn.ReLU(True),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(512, num_classes),
            # torch.nn.ReLU(True),
            # torch.nn.Dropout(0.5),
            # torch.nn.Linear(512, 256),
            # torch.nn.ReLU(True),
            # torch.nn.Dropout(0.5),
            # torch.nn.Linear(256, 128),
            # torch.nn.ReLU(True),
            # torch.nn.Dropout(0.5),
            # torch.nn.Linear(128, 64),
            # torch.nn.ReLU(True),
            # torch.nn.Dropout(0.5),
            # torch.nn.Linear(64, 1),
            torch.nn.Sigmoid(),
            
        )


    def forward(self, x) -> torch.Tensor:
        if self.add_feature > 0:
            y = x[1]
            x = x[0]
        
        x = self.vggnet.features(x)

        x = torch.nn.functional.adaptive_avg_pool2d(x, (7, 7))
        x = torch.flatten(x, 1)

        y = self.vggnet.fc1(y)

        if self.add_feature > 0:
            x = torch.cat((x, y), dim=1)


        x = self.vggnet.classifier(x)


        return x


def VGGNet11(pretrained = False, image_channels = 3, num_classes = 1000, **kwargs) -> VGGNet:

    return VGGNet(image_channels = image_channels, 
                num_classes = num_classes, name='VGGNet11', pretrained = pretrained, **kwargs)
    

def VGGNet13(pretrained = False, image_channels = 3, num_classes = 1000, **kwargs) -> VGGNet:

    return VGGNet(image_channels = image_channels, 
                num_classes = num_classes, name='VGGNet13', pretrained = pretrained,**kwargs)


def VGGNet16(pretrained = False, image_channels = 3, num_classes = 1000, **kwargs) -> VGGNet:

    return VGGNet(image_channels = image_channels, 
                num_classes = num_classes, name='VGGNet16', pretrained = pretrained,**kwargs)

def VGGNet19(pretrained = False, image_channels = 3, num_classes = 1000, **kwargs) -> VGGNet:

    return VGGNet(image_channels = image_channels, 
                num_classes = num_classes, name='VGGNet19', pretrained = pretrained,**kwargs)

   

if __name__ == '__main__':
    model = VGGNet11(pretrained=False, image_channels = 1, num_classes = 1)
    # print(model)

    model.cuda()
    
    inp = torch.randn(1, 1, 500, 625).cuda()
    sx = torch.randn(1, 1).cuda()
    out = model([inp, sx])
    # print(out.argmax(dim=1, keepdim=True))
    print(out)

