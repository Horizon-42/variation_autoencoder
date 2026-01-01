import torch
import torch.nn as nn

class ResizeConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1):
        super().__init__()
        # 1. 上采样：使用 'nearest' (最近邻) 插值是消除棋盘格波纹的关键
        self.upsample = nn.Upsample(scale_factor=stride, mode='nearest')
        # self.upsample = nn.Upsample(scale_factor=stride, mode='bilinear', align_corners=False)
        
        # 2. 卷积：负责学习特征
        # stride=1 保持尺寸不变 (因为尺寸已经由 upsample 变大了)
        # padding_mode='reflection' 消除边缘伪影/黑框
        self.conv = nn.Conv2d(in_channels, out_channels, 
                              kernel_size=kernel_size, 
                              stride=1, 
                              padding=padding, 
                              padding_mode='reflect')

    def forward(self, x):
        x = self.upsample(x)
        x = self.conv(x)
        return x