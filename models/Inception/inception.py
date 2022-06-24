# Deep learning libs
import torch
from torchvision import models


class InceptionV3(torch.nn.Module):
    def __init__(
        self, 
        image_channels: int, 
        num_classes: int = 100, 
        name: str = 'MobileNetV3', 
        pretrained: bool = True,
        input_size: int = 2,
        name_suffix: str = '',
        gender_fc_type: bool = False,
    ) -> None:
        """MobileNetV3 class.

        Args:
            image_channels (int): Number of channels in the input image.
            num_classes (int, optional): Number of classes. Defaults to 100.
            name (str, optional): Name of the model. Defaults to 'MobileNetV2'.
            pretrained (bool, optional): Use pretrained model or not. Defaults to True.
            input_size (int, optional): Input size of the model. Defaults to 2.
        """

        super(InceptionV3, self).__init__()

        self.type = '_Pre' if pretrained else ""
        self.name = name + self.type + name_suffix
        self.in_channels = image_channels
        self.num_classes = num_classes

        self.inception_v3 = models.inception_v3(pretrained=pretrained)

        self.inception_v3.Conv2d_1a_3x3.conv = torch.nn.Conv2d(image_channels, 32, kernel_size=(3, 3), stride=(2, 2), bias=False)

        # Freeze all features to avoid training
        # for param in self.inception_v3.parameters():
        #     param.requires_grad = False

        in_features = self.inception_v3.fc.out_features
        self.add_feature = 0 if input_size <= 1 else input_size - 1

        self.fc_gender = torch.nn.Sequential(

            torch.nn.Linear(1, 32),
            torch.nn.Hardswish(inplace=True),
        )
        self.fc_gender_feature = 32 if gender_fc_type else 1

        self.classifier = torch.nn.Sequential(

            torch.nn.Dropout(0.1),
            torch.nn.Linear(in_features + self.fc_gender_feature, 1000),
            torch.nn.Hardswish(inplace=True),

            torch.nn.Dropout(0.2),
            torch.nn.Linear(1000, 1000),
            torch.nn.Hardswish(inplace=True),

            torch.nn.Dropout(0.1),
            torch.nn.Linear(1000, 512),
            torch.nn.Hardswish(inplace=True),

            torch.nn.Linear(512, num_classes),
            torch.nn.Sigmoid()
        )

    def forward(self, x) -> torch.Tensor:
        if self.add_feature > 0:
            y = x[1]
            x = x[0]
        
        x = self.inception_v3(x)

        if self.fc_gender_feature:
            y = self.fc_gender(y)
        
        x = x.logits
        if self.add_feature > 0:
            x = torch.cat((x, y), dim=1)

        x = self.classifier(x)

        return x


def Inception_V3(**kwargs) -> InceptionV3:
    """
    Constructs a MobileNetV2 model
    Args:
        kwargs: Keyword arguments.
    """
    return InceptionV3(**kwargs)




if __name__ == '__main__':
    model = InceptionV3(pretrained = False, image_channels = 1, num_classes = 1, gender_fc_type = True).cuda()    # print(models.inception_v3(pretrained=False).fc.out_features)
    # print(model.name)
    # print(model.inception_v3.features.parameters().)
    print(model)
    # model.cuda()
    inp = torch.randn(1, 1, 500, 625).cuda()
    sx = torch.randn(1, 1).cuda()
    out = model([inp, sx])
    print(out.shape)
    
