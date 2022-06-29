import torch


"""
Attention blocks
Reference: Learn To Pay Attention
"""
class ProjectorBlock(torch.nn.Module):
    def __init__(self, in_features, out_features):
        super(ProjectorBlock, self).__init__()
        self.conv = torch.nn.Conv2d(in_channels=in_features, out_channels=out_features,
            kernel_size=1, padding=0, bias=False)

    def forward(self, x):
        return self.conv(x)


class SpatialAttn(torch.nn.Module):
    def __init__(self, in_features, normalize_attn=True):
        super(SpatialAttn, self).__init__()
        self.normalize_attn = normalize_attn
        self.conv = torch.nn.Conv2d(in_channels=in_features, out_channels=1,
            kernel_size=1, padding=0, bias=False)

    def forward(self, l, g):
        N, C, H, W = l.size()
        c = self.conv(l+g) # (batch_size,1,H,W)
        if self.normalize_attn:
            a = torch.nn.functional.softmax(c.view(N,1,-1), dim=2).view(N,1,H,W)
        else:
            a = torch.sigmoid(c)
        g = torch.mul(a.expand_as(l), l)
        if self.normalize_attn:
            g = g.view(N,C,-1).sum(dim=2) # (batch_size,C)
        else:
            g = torch.nn.functional.adaptive_avg_pool2d(g, (1,1)).view(N,C)
        return c.view(N,1,H,W), g

