# Deep learning libs
import torch
from torchvision import models, utils
from attention import SpatialAttn, ProjectorBlock

class MobileNetV3(torch.nn.Module):
    def __init__(
        self, 
        image_channels: int, 
        num_classes: int = 100, 
        name: str = 'MobileNetV3_Attention', 
        pretrained: bool = True,
        input_size: int = 2,
        name_suffix: str = '',
        gender_fc_type: bool = False,
        attention: bool = True,
        normalize_attn: bool = True,
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

        self.mobilenet_v3 = models.mobilenet_v3_large(pretrained=pretrained)

        self.mobilenet_v3.features[0][0] = torch.nn.Conv2d(image_channels, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)

        self.attention = attention
        if self.attention:
            self.projector1 = ProjectorBlock(40, 960)
            self.projector2 = ProjectorBlock(80, 960)
            self.projector3 = ProjectorBlock(160, 960)
            self.attn1 = SpatialAttn(in_features=960, normalize_attn=normalize_attn)
            self.attn2 = SpatialAttn(in_features=960, normalize_attn=normalize_attn)
            self.attn3 = SpatialAttn(in_features=960, normalize_attn=normalize_attn)

        self.avgpool = torch.nn.AdaptiveAvgPool2d(1)

        in_features = self.mobilenet_v3.classifier[0].in_features
        self.add_feature = 0 if input_size <= 1 else input_size - 1

        self.fc_gender = torch.nn.Sequential(

            torch.nn.Linear(1, 32),
            torch.nn.Hardswish(inplace=True),
        )
        self.fc_gender_feature = 32 if gender_fc_type else 1

        self.mobilenet_v3.classifier = torch.nn.Sequential(

            # torch.nn.Dropout(0.2),
            torch.nn.Linear((in_features * 3) + self.fc_gender_feature, 2048),
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
        # print("x", x.shape)
        x = self.mobilenet_v3.features[0](x)
        x = self.mobilenet_v3.features[1](x)
        x = self.mobilenet_v3.features[2](x)
        x = self.mobilenet_v3.features[3](x)
        a1 = self.mobilenet_v3.features[4](x)
        
        x = self.mobilenet_v3.features[5](a1)
        x = self.mobilenet_v3.features[6](x)
        x = self.mobilenet_v3.features[7](x)
        a2 = self.mobilenet_v3.features[8](x)

        x = self.mobilenet_v3.features[9](a2)
        x = self.mobilenet_v3.features[10](x)
        x = self.mobilenet_v3.features[11](x)
        x = self.mobilenet_v3.features[12](x)
        x = self.mobilenet_v3.features[13](x)
        a3 = self.mobilenet_v3.features[14](x)

        x = self.mobilenet_v3.features[15](a3)
        x = self.mobilenet_v3.features[16](x)

        x = self.avgpool(x)
        # print("x", x.shape)
        # print("a1", a1.shape)
        # print("a2", a2.shape)
        # print("a3", a3.shape)
        if self.attention:
            m1, x1 = self.attn1(self.projector1(a1), x)
            m2, x2 = self.attn2(self.projector2(a2), x)
            m3, x3 = self.attn3(self.projector3(a3), x)
            # print("1", m1.shape, x1.shape)
            # print("2", m2.shape, x2.shape)
            # print("3", m3.shape, x3.shape)

            x = torch.cat((x1,x2,x3), dim=1) # batch_sizex3C
            # print("x", x.shape)


        x = torch.flatten(x, 1)
        # print("x", x.shape)
        if self.fc_gender_feature:
            y = self.fc_gender(y)

        if self.add_feature > 0:
            x = torch.cat((x, y), dim=1)
        # print("x", x.shape)
        x = self.mobilenet_v3.classifier(x)
        # print("x", x.shape)
        return [x, m1, m2, m3]


def MobileNet_V3_Attention(**kwargs) -> MobileNetV3:
    """
    Constructs a MobileNetV2 model
    Args:
        kwargs: Keyword arguments.
    """
    return MobileNetV3(**kwargs)





if __name__ == '__main__':
    model = MobileNet_V3_Attention(pretrained = True, image_channels = 1, num_classes = 1, gender_fc_type = True).cuda()
    # print(models.mobilenet_v3_large(pretrained=False))
    print(model.name)
    # print(model.mobilenet_v3.features.parameters().)
    print(model)
    # model.cuda()
    inp = torch.randn(1, 1, 500, 500).cuda()
    sx = torch.randn(1, 1).cuda()
    out = model([inp, sx])
    print(out[0].shape, out[1].shape, out[2].shape, out[3].shape)
