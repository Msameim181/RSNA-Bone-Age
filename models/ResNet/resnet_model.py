# Deep learning libs
import torch
from torchsummary import summary


class Block(torch.nn.Module):
    """Convolutional block with Batch Normalization and ReLU activation.

    Args:
        nn (nn.Module): nn.Module package from PyTorch.

    Returns:
        nn.Module: Convolutional block with 3 conv layers and Batch Normalization and ReLU activation.
    """
    def __init__(
        self, in_channels, intermediate_channels, identity_downsample=None, stride=1
    ):
        """Initialization.
        
        Args:
            in_channels (int): Number of input channels.
            intermediate_channels (int): Number of intermediate channels.
            identity_downsample (nn.Module): Downsample layer for identity shortcut.
            stride (int): Stride for the convolution.

        Returns:
            nn.Module: Convolutional block with 3 conv layers and Batch Normalization and ReLU activation.
        """
        super(Block, self).__init__()
        self.expansion = 4
        self.conv1 = torch.nn.Conv2d(
            in_channels, intermediate_channels, kernel_size=1, stride=1, padding=0, bias=False
        )
        self.bn1 = torch.nn.BatchNorm2d(intermediate_channels)
        self.conv2 = torch.nn.Conv2d(
            intermediate_channels,
            intermediate_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False
        )
        self.bn2 = torch.nn.BatchNorm2d(intermediate_channels)
        self.conv3 = torch.nn.Conv2d(
            intermediate_channels,
            intermediate_channels * self.expansion,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False
        )
        self.bn3 = torch.nn.BatchNorm2d(intermediate_channels * self.expansion)
        self.relu = torch.nn.ReLU()
        self.identity_downsample = identity_downsample
        self.stride = stride

    def forward(self, x):
        identity = x.clone()

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)

        if self.identity_downsample is not None:
            identity = self.identity_downsample(identity)

        x += identity
        x = self.relu(x)
        return x


class ResNet(torch.nn.Module):
    """ResNet model.
    
    Args:
        block (nn.Module): Block module.
        layers (list): List of layers.
        image_channels (int): Number of image channels.
        num_classes (int): Number of classes.

    Returns:
        nn.Module: ResNet model.
    
    """

    def __init__(
        self, 
        block, 
        layers: list = None, 
        image_channels: int=3, 
        num_classes: int=100, 
        name: str='ResNet',
        add_nodes: int=1):
        """_summary_

        Args:
            block (Class): From Block class.
            layers (list, optional): Layers list depends on ResNet model. Defaults to [3, 4, 6, 3].
            image_channels (int, optional): Channels of image (RGB:3, Gray:1). Defaults to 3.
            num_classes (int, optional): Class to classification. Defaults to 100.
        """
        if layers is None:
            layers = [3, 4, 6, 3]
        super(ResNet, self).__init__()
        self.name = name
        self.n_channels = image_channels
        self.num_classes = num_classes
        self.in_channels = 64
        self.conv1 = torch.nn.Conv2d(image_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = torch.nn.BatchNorm2d(64)
        self.relu = torch.nn.ReLU()
        self.maxpool = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Essentially the entire ResNet architecture are in these 4 lines below
        self.layer1 = self._make_layer(
            block, layers[0], intermediate_channels=64, stride=1
        )
        self.layer2 = self._make_layer(
            block, layers[1], intermediate_channels=128, stride=2
        )
        self.layer3 = self._make_layer(
            block, layers[2], intermediate_channels=256, stride=2
        )
        self.layer4 = self._make_layer(
            block, layers[3], intermediate_channels=512, stride=2
        )

        self.avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.fc = torch.nn.Linear(512 * 4 + add_nodes, num_classes)

    def forward(self, x):
        y = x[1]
        x = x[0]

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)

        z = x
        y = torch.unsqueeze(y, 1).to(device='cuda', dtype=torch.float32)
        x = torch.cat((z, y), dim=1)

        x = self.fc(x)

        return x

    def _make_layer(self, block, num_residual_blocks, intermediate_channels, stride):

        """

        Args:
            block (Class): From Block class.
            num_residual_blocks (int): Number of residual blocks.
            intermediate_channels (int): Number of intermediate channels.
            stride (int): Stride for the convolution.

        Returns:
            nn.Module: Residual layer.
        """

        identity_downsample = None
        # Either if we half the input space for ex, 56x56 -> 28x28 (stride=2), or channels changes
        # we need to adapt the Identity (skip connection) so it will be able to be added
        # to the layer that's ahead
        if stride != 1 or self.in_channels != intermediate_channels * 4:
            identity_downsample = torch.nn.Sequential(
                torch.nn.Conv2d(
                    self.in_channels,
                    intermediate_channels * 4,
                    kernel_size=1,
                    stride=stride,
                    bias=False
                ),
                torch.nn.BatchNorm2d(intermediate_channels * 4),
            )

        layers = [
            block(
                self.in_channels,
                intermediate_channels,
                identity_downsample,
                stride,
            )
        ]

        # The expansion size is always 4 for ResNet 50,101,152
        self.in_channels = intermediate_channels * 4

        # For example for first resnet layer: 256 will be mapped to 64 as intermediate layer,
        # then finally back to 256. Hence no identity downsample is needed, since stride = 1,
        # and also same amount of channels.
        layers.extend(
            block(self.in_channels, intermediate_channels)
            for _ in range(num_residual_blocks - 1)
        )

        return torch.nn.Sequential(*layers)



# Model sample
def ResNet18(image_channels=3, num_classes=1000):
    return ResNet(Block, [2, 2, 2, 2], image_channels, num_classes, name='ResNet18')

def ResNet34(image_channels=3, num_classes=1000):
    return ResNet(Block, [3, 4, 6, 3], image_channels, num_classes, name='ResNet34')

def ResNet50(image_channels=3, num_classes=1000):
    return ResNet(Block, [3, 4, 6, 3], image_channels, num_classes, name='ResNet50')

def ResNet101(image_channels=3, num_classes=1000):
    return ResNet(Block, [3, 4, 23, 3], image_channels, num_classes, name='ResNet50')

def ResNet152(image_channels=3, num_classes=1000):
    return ResNet(Block, [3, 8, 36, 3], image_channels, num_classes, name='ResNet50')


def test():
    net = ResNet18(image_channels=1, num_classes=229)
    # net.cuda()
    # inp = torch.randn(1, 1, 200, 200).cuda()
    # sx = torch.randn(1).cuda()
    # print(inp.shape)
    # print(sx.shape)
    # # print(inp)
    # # print(sx)
    # out = net([inp, sx])
    # print(net.__class__.__name__)
    # print(net.name)
    # y = net(torch.randn(4, 3, 224, 224)).to("cuda")
    # print(out.size())
    # summary(net, (1, 500, 625), batch_size=1)
    # print(net)

if __name__ == "__main__":
    test()
