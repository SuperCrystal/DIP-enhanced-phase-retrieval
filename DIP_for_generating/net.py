# 网络定义
import numpy as np
import torch
import torch.nn as nn

import torch.nn.functional as F
import torch.utils.data
import torch


class cyl(nn.Module):
    """
    自定义的圆形通光区域
    """
    def __init__(self, d1, d2, N):
        super(cyl, self).__init__()
        # tmp = torch.arange(-d1/2+d1/N, d1/2+d1/N, d1/N)
        tmp = torch.arange(-d1/2, d1/2+d1/(N-1), d1/(N-1))
        x, y = torch.meshgrid(tmp, tmp)
        R = torch.sqrt(x**2 + y**2)
        R = R.cuda()
        self.circle = (R<d2/2) * torch.cuda.FloatTensor([1.0])

    def forward(self, x):
        return torch.mul(x, self.circle)

class conv_block(nn.Module):
    """
    Convolution Block
    """
    def __init__(self, in_ch, out_ch):
        super(conv_block, self).__init__()

        # self.conv = nn.Sequential(
        #     nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
        #     nn.BatchNorm2d(out_ch),
        #     nn.LeakyReLU(inplace=True),
        #     nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
        #     nn.BatchNorm2d(out_ch),
        #     nn.LeakyReLU(inplace=True))

        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(inplace=True))

    def forward(self, x):

        x = self.conv(x)
        return x


class up_conv(nn.Module):
    """
    Up Convolution Block
    """
    def __init__(self, in_ch, out_ch):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x


class U_Net(nn.Module):
    """
    UNet - Basic Implementation
    Paper : https://arxiv.org/abs/1505.04597
    """
    def __init__(self, in_ch=1, out_ch=1):
        super(U_Net, self).__init__()

        n1 = 2    # 16
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]

        self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(in_ch, filters[0])
        self.Conv2 = conv_block(filters[0], filters[1])
        self.Conv3 = conv_block(filters[1], filters[2])
        self.Conv4 = conv_block(filters[2], filters[3])
        self.Conv5 = conv_block(filters[3], filters[4])

        self.Up5 = up_conv(filters[4], filters[3])
        self.Up_conv5 = conv_block(filters[4], filters[3])

        self.Up4 = up_conv(filters[3], filters[2])
        self.Up_conv4 = conv_block(filters[3], filters[2])

        self.Up3 = up_conv(filters[2], filters[1])
        self.Up_conv3 = conv_block(filters[2], filters[1])

        self.Up2 = up_conv(filters[1], filters[0])
        # self.Up_conv2 = conv_block(filters[1], filters[0])

        # self.Conv = nn.Conv2d(filters[0], out_ch, kernel_size=1, stride=1, padding=0)
        self.Conv = nn.Conv2d(filters[1], out_ch, kernel_size=1, stride=1, padding=0)
        self.active = torch.nn.Sigmoid()
        # self.active0 = torch.nn.Sigmoid()
        self.active1 = torch.nn.Sigmoid()
        self.active2 = torch.nn.Sigmoid()
        self.active3 = torch.nn.Sigmoid()

        # self.defocus_param0 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.defocus_param1 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.defocus_param2 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.defocus_param3 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)

        self.cyl = cyl(34.31, 10, 1024)

    def forward(self, x, defocus_term):

        e1 = self.Conv1(x)

        e2 = self.Maxpool1(e1)
        e2 = self.Conv2(e2)

        e3 = self.Maxpool2(e2)
        e3 = self.Conv3(e3)

        e4 = self.Maxpool3(e3)
        e4 = self.Conv4(e4)

        e5 = self.Maxpool4(e4)
        e5 = self.Conv5(e5)

        d5 = self.Up5(e5)
        d5 = torch.cat((e4, d5), dim=1)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((e3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((e2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((e1, d2), dim=1)
        # d2 = self.Up_conv2(d2)

        out = self.Conv(d2)

        out = self.active(out)
        # out = out * 2 * np.pi * 0.5
        # out = out * 2 * np.pi * 0.5 - np.pi/2  # -np.pi使得挖空部分可以被检测到？
        out = out * 2 * np.pi * 1

        # 支持域？
        out = self.cyl(out)

        # param0_limited = self.active0(self.defocus_param1) * 0.000001
        # param0_limited = 0
        param1_limited = self.active1(self.defocus_param1) * 0.2
        param2_limited = self.active2(self.defocus_param2) * 0.3
        param3_limited = self.active3(self.defocus_param3) * 0.5

        # d0 = torch.mul(defocus_term, param0_limited)
        d1 = torch.mul(defocus_term, param1_limited)
        d2 = torch.mul(defocus_term, param2_limited)
        d3 = torch.mul(defocus_term, param3_limited)
        # d1 = torch.mul(defocus_term, self.defocus_param1)
        # d2 = torch.mul(defocus_term, self.defocus_param2)
        # d3 = torch.mul(defocus_term, self.defocus_param3)
        # out0 = torch.add(out, d0)
        out0 = out
        out1 = torch.add(out, d1)   # 这里用了out的值
        out2 = torch.add(out, d2)
        out3 = torch.add(out, d3)
        return out0, out1, out2, out3, param1_limited, param2_limited, param3_limited
        # return out, out1, out2, out3, self.defocus_param1, self.defocus_param2, self.defocus_param3
