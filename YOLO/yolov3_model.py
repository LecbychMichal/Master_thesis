from torchsummary import summary
import torch.nn as nn
import torch

""""
Implementation of official YOLOv3
"""

class Conv(nn.Module):
    def __init__(self, in_channels, out_channels, bn_act=True, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                              bias=not bn_act, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels)
        self.leaky = nn.LeakyReLU(0.1)
        self.use_bn_act = bn_act

    def forward(self, x):
        if self.use_bn_act:
            return self.leaky(self.bn(self.conv(x)))
        else:
            return self.conv(x)


class ResBlock(nn.Module):
    def __init__(self, channels, use_residual=True, num_repeats=1):
        super().__init__()
        self.layers = nn.ModuleList()
        for repeat in range(num_repeats):
            self.layers += [
                nn.Sequential(
                    Conv(channels, channels // 2, kernel_size=1),
                    Conv(channels // 2, channels,
                         kernel_size=3, padding=1),
                )
            ]

        self.use_residual = use_residual
        self.num_repeats = num_repeats

    def forward(self, x):
        for layer in self.layers:
            if self.use_residual:
                x = x + layer(x)
            else:
                x = layer(x)

        return x


class Scale(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.pred = nn.Sequential(
            Conv(in_channels, 2 * in_channels, kernel_size=3, padding=1),
            Conv(
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


class YOLOv3(nn.Module):
    def __init__(self, in_channels=3, num_classes=20):
        super().__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.depths = [64, 128, 256, 512, 1024]
        self.num_repeats = [1, 2, 8, 8, 4]
        self.up = [512, 256, 128]

        start = nn.Sequential(
            Conv(in_channels, 32, kernel_size=3, stride=1, padding=1),
            Conv(32, 64, kernel_size=3, stride=2, padding=1),
        )

        self.downsample_layers = nn.ModuleList()
        self.downsample_layers.append(start)
        for i in range(4):
            downsample = nn.Sequential(
                Conv(self.depths[i], self.depths[i+1],
                     kernel_size=3, stride=2, padding=1),
            )
            self.downsample_layers.append(downsample)

        self.stages = nn.ModuleList()
        for i in range(5):
            stage = nn.Sequential(
                ResBlock(channels=self.depths[i], use_residual=True,
                         num_repeats=self.num_repeats[i])
            )

            self.stages.append(stage)

        scale1 = Scale(512, num_classes=self.num_classes)
        scale2 = Scale(256, num_classes=self.num_classes)
        scale3 = Scale(128, num_classes=self.num_classes)
        self.scales = nn.ModuleList()
        self.scales.append(scale1)
        self.scales.append(scale2)
        self.scales.append(scale3)

        route1 = nn.Sequential(
            Conv(1024, 512, kernel_size=1, stride=1, padding=0),
            Conv(512, 1024, kernel_size=3, stride=1, padding=1),
            ResBlock(1024, use_residual=False),
            Conv(1024, 512, kernel_size=1),
        )

        route2 = nn.Sequential(
            Conv(768, 256, kernel_size=1, stride=1, padding=0),
            Conv(256, 512, kernel_size=3, stride=1, padding=1),
            ResBlock(512, use_residual=False),
            Conv(512, 256, kernel_size=1),
        )

        route3 = nn.Sequential(
            Conv(384, 128, kernel_size=1, stride=1, padding=0),
            Conv(128, 256, kernel_size=3, stride=1, padding=1),
            ResBlock(256, use_residual=False),
            Conv(256, 128, kernel_size=1),
        )
        self.routes = nn.ModuleList()
        self.routes.append(route1)
        self.routes.append(route2)
        self.routes.append(route3)

        self.upsample = nn.ModuleList()
        for i in range(2):
            upsample = nn.Sequential(
                Conv(self.up[i], self.up[i+1],
                     kernel_size=1, stride=1, padding=0),
                nn.Upsample(scale_factor=2)
            )
            self.upsample.append(upsample)

    def forward(self, x):
        """¨¨¨¨¨¨¨¨¨¨DARKNET-53¨¨¨¨¨¨¨¨¨¨¨"""
        routes = []
        for i in range(5):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
            if self.num_repeats[i] == 8:
                routes.append(x)
        """¨¨¨¨¨¨¨¨¨¨DARKNET-53¨¨¨¨¨¨¨¨¨¨¨"""
        outs = []

        for i in range(2):
            x = self.routes[i](x)
            outs.append(self.scales[i](x))
            x = self.upsample[i](x)
            x = torch.cat([x, routes[-i-1]], dim=1)

        x = self.routes[-1](x)
        outs.append(self.scales[-1](x))

        return outs
