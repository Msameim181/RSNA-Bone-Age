# Deep learning libs
import torch
from torchvision import models, utils
from .attention import SpatialAttn, ProjectorBlock

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
            self.projector1 = ProjectorBlock(24, 960)
            self.projector2 = ProjectorBlock(40, 960)
            self.projector3 = ProjectorBlock(112, 960)
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
        a1 = self.mobilenet_v3.features[2](x)

        x = self.mobilenet_v3.features[3](a1)
        x = self.mobilenet_v3.features[4](x)
        x = self.mobilenet_v3.features[5](x)
        a2 = self.mobilenet_v3.features[6](x)

        x = self.mobilenet_v3.features[7](a2)
        x = self.mobilenet_v3.features[8](x)
        x = self.mobilenet_v3.features[9](x)
        x = self.mobilenet_v3.features[10](x)
        x = self.mobilenet_v3.features[11](x)
        a3 = self.mobilenet_v3.features[12](x)

        x = self.mobilenet_v3.features[13](a3)
        x = self.mobilenet_v3.features[14](x)
        x = self.mobilenet_v3.features[15](x)
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




# import cv2
# def visualize_attn(I, c):
#     # Image
#     img = I.permute((1,2,0)).cpu().numpy()
#     # Heatmap
#     N, C, H, W = c.size()
#     a = torch.nn.functional.softmax(c.view(N,C,-1), dim=2).view(N,C,H,W)
#     up_factor = 32/H
#     print(up_factor, I.size(), c.size())
#     if up_factor > 1:
#         a = torch.nn.functional.interpolate(a, scale_factor=up_factor, mode='bilinear', align_corners=False)
#     attn = utils.make_grid(a, nrow=4, normalize=True, scale_each=True)
#     attn = attn.permute((1,2,0)).mul(255).byte().cpu().numpy()
#     attn = cv2.applyColorMap(attn, cv2.COLORMAP_JET)
#     attn = cv2.cvtColor(attn, cv2.COLOR_BGR2RGB)
#     # Add the heatmap to the image
#     vis = 0.4 * attn
#     return torch.from_numpy(vis).permute(2,0,1)

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
    # print(out.shape)

    # import matplotlib.pyplot as plt
    # model.eval()
    # with torch.no_grad():
    #     images = torch.randn(1, 1, 500, 500).cuda()
    #     sx = torch.randn(1, 1).cuda()
    #     I = utils.make_grid(images, nrow=4, normalize=True, scale_each=True)
    #     # writer.add_image('origin', I)
    #     _, c1, c2, c3 = model([images, sx])
    #     # print(I.shape, c1.shape, c2.shape, c3.shape, c4.shape)
    #     attn1 = visualize_attn(I, c1)
    #     # writer.add_image('attn1', attn1)
    #     attn2 = visualize_attn(I, c2)
    #     # writer.add_image('attn2', attn2)
    #     attn3 = visualize_attn(I, c3)
    #     # writer.add_image('attn3', attn3)
        
    #     # plot image and attention maps
    #     plt.figure(figsize=(10, 10))
    #     plt.subplot(2, 4, 1)
    #     plt.imshow(I.permute((1,2,0)).cpu().numpy())
    #     plt.subplot(2, 4, 2)
    #     plt.imshow(utils.make_grid(c1, nrow=4).permute(1, 2, 0).cpu().numpy())
    #     plt.subplot(2, 4, 3)
    #     plt.imshow(utils.make_grid(c2, nrow=4).permute(1, 2, 0).cpu().numpy())
    #     plt.subplot(2, 4, 4)
    #     plt.imshow(utils.make_grid(c3, nrow=4).permute(1, 2, 0).cpu().numpy())
    #     plt.subplot(2, 4, 5)
    #     plt.imshow(attn1.permute((1,2,0)).cpu().numpy())
    #     plt.subplot(2, 4, 4)
    #     plt.imshow(attn2.permute((1,2,0)).cpu().numpy())
    #     plt.subplot(2, 4, 7)
    #     plt.imshow(attn3.permute((1,2,0)).cpu().numpy())
    #     plt.show()

    
