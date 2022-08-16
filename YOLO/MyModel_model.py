from torchsummary import summary
import torch.nn as nn
import torch
import torch.nn.functional as F

""""
Implementation of my object detection model - the model implementation is influenced by paper A ConvNet for the 2020s
"""

class My_Conv(nn.Module):
    def __init__(self, in_channels, out_channels, bn_act=True, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                              bias=not bn_act, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels)
        self.leaky = My_Mish()
        self.use_bn_act = bn_act

    def forward(self, x):
        if self.use_bn_act:
            return self.leaky(self.bn(self.conv(x)))
        else:
            return self.conv(x)


class My_Mish(nn.Module):
    def __init__(self):
        super(My_Mish, self).__init__()

    def forward(self, x):
        return x * torch.tanh(F.softplus(x))


class My_ConvBlock(nn.Module):

    def __init__(self, dim, drop_path=0.0, layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(
            dim, dim, kernel_size=7, padding=3, groups=dim
        )  # depthwise conv
        self.norm = nn.BatchNorm2d(dim)
        self.pwconv1 = nn.Linear(
            dim, 4 * dim
        )  # pointwise/1x1 convs, implemented with linear layers
        self.act = My_Mish()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = (
            nn.Parameter(layer_scale_init_value *
                         torch.ones((dim)), requires_grad=True)
            if layer_scale_init_value > 0
            else None
        )
        self.drop_path = nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x


class My_Scale(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.pred = nn.Sequential(
            My_Conv(in_channels, 2 * in_channels, kernel_size=3, padding=1),
            My_Conv(
                2 * in_channels, (num_classes + 5) * 3, bn_act=False, kernel_size=1
            ),
        )
        self.num_classes = num_classes

    def forward(self, x):
        return (
            self.pred(x)
            .reshape(x.shape[0], 3, self.num_classes + 5, x.shape[2], x.shape[3])
            .permute(0, 1, 3, 4, 2)
        )


class My_SpatialPyramidPooling(nn.Module):
    def __init__(self, feature_channels, pool_sizes=[5, 9, 13]):
        super(My_SpatialPyramidPooling, self).__init__()

        self.head_conv = nn.Sequential(
            My_Conv(feature_channels[-1],
                 feature_channels[-1] // 2, kernel_size=1),
        )

        self.maxpools = nn.ModuleList(
            [
                nn.MaxPool2d(pool_size, 1, pool_size // 2)
                for pool_size in pool_sizes
            ]
        )
        self.__initialize_weights()

    def forward(self, x):
        x = self.head_conv(x)
        features = [maxpool(x) for maxpool in self.maxpools]
        features = torch.cat([x] + features, dim=1)

        return features

    def __initialize_weights(self):

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.01)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class My_ResBlock(nn.Module):
    def __init__(self, channels, use_residual=True, num_repeats=1):
        super().__init__()
        self.layers = nn.ModuleList()
        for repeat in range(num_repeats):
            self.layers += [
                nn.Sequential(
                    My_ConvBlock(channels)
                )
            ]

        self.use_residual = use_residual
        self.num_repeats = num_repeats

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class MyModel(nn.Module):
    def __init__(self, in_channels=3, num_classes=20):
        super().__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.depths = [64, 128, 256, 512, 1024]
        self.num_repeats = [1, 2, 5, 5, 4]
        self.up = [512, 256, 128]
        self.feature_channels = [256, 512, 1024]

        start = nn.Sequential(
            My_Conv(in_channels, 32, kernel_size=3, stride=1, padding=1),
            My_Conv(32, 64, kernel_size=3, stride=2, padding=1),
        )

        self.downsample_layers = nn.ModuleList()
        self.downsample_layers.append(start)
        for i in range(4):
            downsample = nn.Sequential(
                My_Conv(self.depths[i], self.depths[i+1],
                     kernel_size=3, stride=2, padding=1),
            )
            self.downsample_layers.append(downsample)

        self.stages = nn.ModuleList()
        for i in range(5):
            stage = nn.Sequential(
                My_ResBlock(channels=self.depths[i], use_residual=True,
                         num_repeats=self.num_repeats[i])
            )

            self.stages.append(stage)

        scale1 = My_Scale(512, num_classes=self.num_classes)
        scale2 = My_Scale(256, num_classes=self.num_classes)
        scale3 = My_Scale(128, num_classes=self.num_classes)
        self.scales = nn.ModuleList()
        self.scales.append(scale1)
        self.scales.append(scale2)
        self.scales.append(scale3)

        route1 = nn.Sequential(
            My_Conv(1024, 1024, kernel_size=3, stride=1, padding=1),
            My_ResBlock(1024, use_residual=False),
            My_Conv(1024, 512, kernel_size=1),
        )

        route2 = nn.Sequential(
            My_Conv(768, 256, kernel_size=1, stride=1, padding=0),
            My_Conv(256, 512, kernel_size=3, stride=1, padding=1),
            My_ResBlock(512, use_residual=False),
            My_Conv(512, 256, kernel_size=1),
        )

        route3 = nn.Sequential(
            My_Conv(384, 128, kernel_size=1, stride=1, padding=0),
            My_Conv(128, 256, kernel_size=3, stride=1, padding=1),
            My_ResBlock(256, use_residual=False),
            My_Conv(256, 128, kernel_size=1),
        )
        self.routes = nn.ModuleList()
        self.routes.append(route1)
        self.routes.append(route2)
        self.routes.append(route3)

        self.upsample = nn.ModuleList()
        for i in range(2):
            upsample = nn.Sequential(
                My_Conv(self.up[i], self.up[i+1],
                     kernel_size=1, stride=1, padding=0),
                nn.Upsample(scale_factor=2)
            )
            self.upsample.append(upsample)

        self.spp = My_SpatialPyramidPooling(self.feature_channels)
        self.downstream_conv = nn.Sequential(
            My_Conv(2048, 1024, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x):
        """¨¨¨¨¨¨¨¨¨¨BACKBONE¨¨¨¨¨¨¨¨¨¨¨"""
        routes = []
        for i in range(5):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
            if self.num_repeats[i] == 5:
                routes.append(x)
        """¨¨¨¨¨¨¨¨¨¨BACKBONE¨¨¨¨¨¨¨¨¨¨¨"""

        """-----------NECK---------------"""
        x = self.spp(x)
        x = self.downstream_conv(x)
        """-----------NECK--------------"""

        """-----------HEAD-------------"""

        outs = []
        for i in range(2):
            x = self.routes[i](x)
            outs.append(self.scales[i](x))
            x = self.upsample[i](x)
            x = torch.cat([x, routes[-i-1]], dim=1)

        x = self.routes[-1](x)
        outs.append(self.scales[-1](x))
        """-----------HEAD-------------"""

        return outs

