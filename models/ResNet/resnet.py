from typing import Any, Callable, List, Optional, Type, Union

# Deep learning libs
import torch
from torchsummary import summary
import torchvision.models as models


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> torch.nn.Conv2d:
    """3x3 convolution with padding"""
    return torch.nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> torch.nn.Conv2d:
    """1x1 convolution"""
    return torch.nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class BasicBlock(torch.nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[torch.nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., torch.nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = torch.nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = torch.nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(torch.nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[torch.nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., torch.nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = torch.nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = torch.nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(torch.nn.Module):
    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        image_channels: int = 3,
        num_classes: int = 1000,
        add_last_layer_node: int = 1, 
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., torch.nn.Module]] = None,
        name: str='ResNet',
    ) -> None:
        super().__init__()

        self.name = name
        self.in_channels = image_channels
        self.num_classes = num_classes

        if norm_layer is None:
            norm_layer = torch.nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                f"or a 3-element tuple, got {replace_stride_with_dilation}"
            )
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = torch.nn.Conv2d(self.in_channels, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = torch.nn.ReLU(inplace=True)
        self.maxpool = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])
        self.avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.fc = torch.nn.Linear(512 * block.expansion + add_last_layer_node, self.num_classes)

        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (torch.nn.BatchNorm2d, torch.nn.GroupNorm)):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    torch.nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock):
                    torch.nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        planes: int,
        blocks: int,
        stride: int = 1,
        dilate: bool = False,
    ) -> torch.nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = torch.nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )

        return torch.nn.Sequential(*layers)

    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        y = x[1]
        x = x[0]

        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        z = x
        y = torch.unsqueeze(y, 1).to(device='cuda', dtype=torch.float32)
        x = torch.cat((z, y), dim=1)

        x = self.fc(x)

        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._forward_impl(x)

class ResNet_Pre(torch.nn.Module):
    def __init__(
        self,
        image_channels: int = 3,
        num_classes: int = 1000,
        name: str='ResNet',
        **kwargs
    ) -> None:
        super(ResNet_Pre, self).__init__()


        self.name = name
        self.in_channels = image_channels
        self.num_classes = num_classes
        self.inplanes = 64

        if name == "ResNet18":
            self.resnet = models.resnet18(pretrained=True)
        elif name == "ResNet34":
            self.resnet = models.resnet34(pretrained=True)
        elif name == "ResNet50":
            self.resnet = models.resnet50(pretrained=True)
        elif name == "ResNet101":
            self.resnet = models.resnet101(pretrained=True)

        self.resnet.conv1 = torch.nn.Conv2d(self.in_channels, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)

        self.fc = torch.nn.Sequential(
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(1001, 512),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(512, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x[1]
        x = x[0]
        
        x = self.resnet(x)

        z = x
        y = torch.unsqueeze(y, 1).to(device='cuda', dtype=torch.float32)
        x = torch.cat((z, y), dim=1)

        x = self.fc(x)


        return x






def ResNet18(pretrained = False, image_channels = 3, num_classes = 1000, **kwargs) -> ResNet:
    if pretrained:
        return ResNet_Pre(image_channels = image_channels, 
                    num_classes = num_classes, name='ResNet18', **kwargs)
    return ResNet(BasicBlock, [2, 2, 2, 2], 
                    image_channels = image_channels, 
                    num_classes = num_classes, name='ResNet18', **kwargs)

def ResNet34(pretrained = False, image_channels = 3, num_classes = 1000, **kwargs) -> ResNet:
    if pretrained:
        return ResNet_Pre(image_channels = image_channels, 
                    num_classes = num_classes, name='ResNet34', **kwargs)
    return ResNet(BasicBlock, [3, 4, 6, 3], 
                    image_channels = image_channels, 
                    num_classes = num_classes, name='ResNet34', **kwargs)

def ResNet50(pretrained = False, image_channels = 3, num_classes = 1000, **kwargs) -> ResNet:
    if pretrained:
        return ResNet_Pre(image_channels = image_channels, 
                    num_classes = num_classes, name='ResNet50', **kwargs)
    return ResNet(Bottleneck, [3, 4, 6, 3], 
                    image_channels = image_channels, 
                    num_classes = num_classes, name='ResNet50', **kwargs)

def ResNet101(pretrained = False, image_channels = 3, num_classes = 1000, **kwargs) -> ResNet:
    if pretrained:
        return ResNet_Pre(image_channels = image_channels, 
                    num_classes = num_classes, name='ResNet101', **kwargs)
    return ResNet(Bottleneck, [3, 4, 23, 3], 
                    image_channels = image_channels, 
                    num_classes = num_classes, name='ResNet101', **kwargs)

def ResNet152(pretrained = False, image_channels = 3, num_classes = 1000, **kwargs) -> ResNet:
    if pretrained:
        return ResNet_Pre(image_channels = image_channels, 
                    num_classes = num_classes, name='ResNet152', **kwargs)
    return ResNet(Bottleneck, [3, 8, 36, 3], 
                    image_channels = image_channels, 
                    num_classes = num_classes, name='ResNet152', **kwargs)




if __name__ == '__main__':
    model = ResNet18(pretrained = True, image_channels = 1, num_classes = 229).cuda()
    print(model)
    print(model.name)
    # input_shape = (1, 500, 625)
    # # summary(model, input_shape, batch_size=1)
    # inp = torch.randn(1, 1, 500, 625).cuda()
    # sx = torch.randn(1).cuda()
    
    # out = model([inp, sx])
    # print(out.shape)
