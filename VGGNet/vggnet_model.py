"""
A from scratch implementation of the VGG architecture.
"""

# Imports
import torch
import torch.nn as nn  # All neural network modules, nn.Linear, nn.Conv2d, BatchNorm, Loss functions

VGG_types = {
    "VGG11": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "VGG13": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "VGG16": [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"],
    "VGG19": [64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, 512, "M", 512, 512, 512, 512, "M"],
}


class VGG_net(nn.Module):
    def __init__(self, in_channels=3, num_classes=1000, type="VGG11"):
        super(VGG_net, self).__init__()
        self.name = type
        self.n_channels = in_channels
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.conv_layers = self.create_conv_layers(VGG_types[type])

        self.fcs = nn.Sequential(
            nn.Linear(512 * 285, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
        )
        self.fc = nn.Linear(4097, num_classes)

    def forward(self, x):
        y = x[1]
        x = x[0]
        x = self.conv_layers(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fcs(x)

        z = x
        y = torch.unsqueeze(y, 1).to(device='cuda', dtype=torch.float32)
        z = torch.cat((z, y), dim=1)
        
        x = self.fc(z)
        return x

    def create_conv_layers(self, architecture):
        layers = []
        in_channels = self.in_channels

        for x in architecture:
            if type(x) == int:
                out_channels = x

                layers += [
                    nn.Conv2d(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=(3, 3),
                        stride=(1, 1),
                        padding=(1, 1),
                    ),
                    nn.BatchNorm2d(x),
                    nn.ReLU(),
                ]
                in_channels = x
            elif x == "M":
                layers += [nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))]

        return nn.Sequential(*layers)


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    net = VGG_net(in_channels=1, num_classes=229, type="VGG11").to(device)
    # print(net)
    inp = torch.randn(1, 1, 500, 625).cuda()
    sx = torch.randn(1).cuda()

    out = net([inp, sx])
    # N = 3 (Mini batch size)
    # x = torch.randn(1, 3, 224, 224).to(device)
    # print(net(x).shape)