'''
Based on https://github.com/qingOOyuan/mini_Xception/blob/main/utils/Model.py and https://github.com/otaha178/Emotion-recognition/blob/master/models/cnn.py
'''
import torch.nn as nn

class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, bias=False, weight_decay=0.01):
        super(SeparableConv2d, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, padding=padding, groups=in_channels,
                                   bias=bias, padding_mode='zeros')
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)
        self._init_weights(weight_decay)

    def _init_weights(self, weight_decay):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                m.weight_decay = weight_decay

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

class MiniXception(nn.Module):
    def __init__(self, num_classes=7, in_channels=1, l2_reg=0.01):
        super(MiniXception, self).__init__()
        self.reg = l2_reg

        def conv_bn_relu(in_channels, out_channels, kernel_size=3, stride=1):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU()
            )

        def sep_module(in_ch, out_ch):
            return nn.Sequential(
                SeparableConv2d(in_ch, out_ch, weight_decay=self.reg),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(),
                SeparableConv2d(out_ch, out_ch, weight_decay=self.reg),
                nn.BatchNorm2d(out_ch)
            )

        def residual_block(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=2, padding=0, bias=False),
                nn.BatchNorm2d(out_ch)
            )

        self.conv1 = conv_bn_relu(in_channels, 8)
        self.conv2 = conv_bn_relu(8, 8)

        self.block1 = sep_module(8, 16)
        self.res1 = residual_block(8, 16)

        self.block2 = sep_module(16, 32)
        self.res2 = residual_block(16, 32)

        self.block3 = sep_module(32, 64)
        self.res3 = residual_block(32, 64)

        self.block4 = sep_module(64, 128)
        self.res4 = residual_block(64, 128)

        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.final_conv = nn.Conv2d(128, num_classes, kernel_size=3, padding=1)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)

        r = self.res1(x)
        x = self.block1(x)
        x = self.pool(x)
        x = x + r

        r = self.res2(x)
        x = self.block2(x)
        x = self.pool(x)
        x = x + r

        r = self.res3(x)
        x = self.block3(x)
        x = self.pool(x)
        x = x + r

        r = self.res4(x)
        x = self.block4(x)
        x = self.pool(x)
        x = x + r

        x = self.final_conv(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        return x
