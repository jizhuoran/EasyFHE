from torch.hub import load_state_dict_from_url
import torch
import torch.nn as nn
from typing import Any, Callable, List, Optional, Type, Union
from torch import Tensor
from math import pi, sqrt
from torch.autograd import Function

def get_resnet20_HerPN(num_classes):
    return ResNet20_HerPN(block=BasicBlock_HerPN,num_classes=num_classes)

class ResNet18_HerPN(nn.Module):

    def __init__(self, block, num_classes=10):
        super(ResNet18_HerPN, self).__init__()
        self.inplanes = 64
        self.conv1 = conv3x3(3, 64)
        self.HerPN1 = HerPN2d(64)
        self.layer1 = self._make_layer(block, 64, 2)
        self.layer2 = self._make_layer(block, 128, 2, stride=2)
        self.layer3 = self._make_layer(block, 256, 2, stride=2)
        self.layer4 = self._make_layer(block, 512, 2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')


    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x,fea_out = False):
        if fea_out:
            fea = []
            x = self.conv1(x)
            x = self.HerPN1(x)
            fea.append(x)

            x = self.layer1(x)
            fea.append(x)

            x = self.layer2(x)
            fea.append(x)

            x = self.layer3(x)
            fea.append(x)
            x = self.layer4(x)
            fea.append(x)
            x = self.avgpool(x)
            x = x.view(x.size(0), -1)
            x = self.fc(x)
            return x,fea
        else:

            x = self.conv1(x)
            x = self.HerPN1(x)
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)
            x = self.avgpool(x)
            x = x.view(x.size(0), -1)
            x = self.fc(x)
            return x

def get_resnet18_HerPN(num_classes):
    return ResNet18_HerPN(block=BasicBlock_HerPN,num_classes=num_classes)


class BasicBlock_HerPN(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.HerPN1 = HerPN2d(num_features=planes)
        self.conv2 = conv3x3(planes, planes)
        self.downsample = downsample
        self.HerPN2 = HerPN2d(num_features=planes)
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)
        out = self.conv1(x)
        print('max conv1',torch.max(out))
        print('min conv1', torch.min(out))
        # print('conv1',torch.max(out))
        out = self.HerPN1(out)
        print('max HerPN1 ',torch.max(out))
        print('min HerPN1', torch.min(out))
        # print('herPN',torch.max(out))
        out = self.conv2(out)
        print('max conv2',torch.max(out))
        print('min conv2', torch.min(out))
        # print('conv2', torch.max(out))
        out += identity
        # print('sum', torch.max(out))
        out = self.HerPN2(out)
        print('max HerPN2',torch.max(out))
        print('min HerPN2', torch.min(out))

        return out

class ResNet20_HerPN(nn.Module):

    def __init__(self, block, num_classes=10):
        super(ResNet20_HerPN, self).__init__()
        self.inplanes = 16
        self.conv1 = conv3x3(3, 16)
        self.HerPN1 = HerPN2d(16)
        self.layer1 = self._make_layer(block, 16, 3)
        self.layer2 = self._make_layer(block, 32, 3, stride=2)
        self.layer3 = self._make_layer(block, 64, 3, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')


    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x, fea_out=False):
        if fea_out:
            fea = []
            x = self.conv1(x)
            x = self.HerPN1(x)
            fea.append(x)

            x = self.layer1(x)
            fea.append(x)

            x = self.layer2(x)
            fea.append(x)

            x = self.layer3(x)
            fea.append(x)

            x = self.avgpool(x)
            x = x.view(x.size(0), -1)
            x = self.fc(x)
            return x, fea
        else:

            x = self.conv1(x)
            x = self.HerPN1(x)

            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)

            x = self.avgpool(x)
            x = x.view(x.size(0), -1)
            x = self.fc(x)
            return x

def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )

def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class HerPN2d(nn.Module):
    """
    三个埃尔米特多项式基底hi（x）:1，x，x^2-1,归一化系数，1，1，0.707
    对应f系数：1 / sqrt(2 * pi), 1 / 2, 1 / sqrt(4 * pi)
    """
    @staticmethod
    def h0(x):
        return torch.ones(x.shape).to(x.device)

    @staticmethod
    def h1(x):
        return x

    @staticmethod
    def h2(x):
        return (x * x - 1)  * 0.7071

    def __init__(self, num_features: int, BN_dimension=2):
        super().__init__()
        self.f = (1 / sqrt(2 * pi), 1 / 2, 1 / sqrt(4 * pi))
        # self.f = ( 1 / 2, 1 / sqrt(4 * pi))
        self.num_channels = num_features
        # 可学习的缩放和偏移参数
        self.gamma = nn.Parameter(torch.ones(self.num_channels),requires_grad=True)  # 形状：(num_channels,)
        self.beta = nn.Parameter(torch.zeros(self.num_channels),requires_grad=True)  # 形状：(num_channels,)


        if (BN_dimension == 1):
            self.bn0 = nn.BatchNorm1d(num_features, affine=False)
            self.bn1 = nn.BatchNorm1d(num_features, affine=False)
            self.bn2 = nn.BatchNorm1d(num_features, affine=False)
        else:
            self.bn0 = nn.BatchNorm2d(num_features, affine=False)
            self.bn1 = nn.BatchNorm2d(num_features, affine=False)
            self.bn2 = nn.BatchNorm2d(num_features, affine=False)

        self.bn = (self.bn0, self.bn1, self.bn2)
        # self.bn = ( self.bn1, self.bn2)
        self.h = (self.h0, self.h1, self.h2)
        # self.h = (self.h1, self.h2)

    def forward(self, x):
        result = torch.zeros(x.shape).to(x.device)

        for bn, f, h in zip(self.bn, self.f, self.h):
            temp = h(x)
            temp = bn(temp)
            temp = torch.mul(f, temp)
            result = torch.add(result, temp)
        result = self.gamma.view(1, -1, 1, 1) * result + self.beta.view(1, -1, 1, 1)
        return result

class MultiChannelPAF(nn.Module):
    def __init__(self,init_a2, init_a1, init_a0):
        """
        多通道二阶激活函数模块
        参数：
            num_channels: 通道数
            init_a2, init_a1, init_a0: 参数的初始值，默认为 1.0, 0.0, 0.0
        """
        super(MultiChannelPAF, self).__init__()
        # 将 a2, a1, a0 定义为可学习的参数
        self.a2 = init_a2
        self.a1 = init_a1
        self.a0 = init_a0
    def forward(self, x):
        """
        前向传播
        参数：
            x: 输入张量，形状为 (batch_size, num_channels, height, width)
        返回：
            输出张量，形状与 x 相同
        """
        return MultiChannelPoloActFunction.apply(x, self.a2, self.a1, self.a0)

class MultiChannelPoloActFunction(Function):
    @staticmethod
    def forward(ctx, input, a2, a1, a0):
        """
        前向传播：计算 y = a2 * x^2 + a1 * x + a0
        参数：
            input: 输入张量，形状为 (batch_size, num_channels, height, width)
            a2, a1, a0: 参数张量，形状为 (num_channels,)
        返回：
            输出张量，形状与 input 相同
        """
        # 保存输入和参数，以便在反向传播时使用
        ctx.save_for_backward(input, a2, a1, a0)
        # print(input)


        # 将 a2, a1, a0 扩展为与 input 相同的形状
        a2 = a2.view(1, -1, 1, 1)  # 形状变为 (1, num_channels, 1, 1)
        a1 = a1.view(1, -1, 1, 1)  # 形状变为 (1, num_channels, 1, 1)
        a0 = a0.view(1, -1, 1, 1)  # 形状变为 (1, num_channels, 1, 1)
        part1= a1 * input
        part2 = a2 * input.pow(2)

        # 计算正向传播
        output = part2 + part1 + a0
        return output

def change_HerPN2d_into_PAF_MutalChannel(model:HerPN2d):
    """
    本函数用于将一个HerPN2d转为一个多通道的PAF：
    三个基底：h0可以完全去除;h1与h2正常处理
    :param model:
    :return:
    """
    bn1 = model.bn1
    bn2 = model.bn2
    gamma = model.gamma
    beta=model.beta
    var2 = (bn2.running_var + 1e-05)**-0.5
    var1 = (bn1.running_var + 1e-05)**-0.5
    u2 = bn2.running_mean
    u1 = bn1.running_mean
    w2 = gamma * var2 / sqrt(4 * pi)
    w1 = gamma * var1 * 0.5
    a2 = 0.5 * sqrt(2) * w2
    a1 = w1
    a0 = beta - 0.5 * sqrt(2) * w2 - u2 * w2 - w1 * u1
    mask = a2 < 0
    modified_count = mask.sum().item()
    a2[mask] = 1e-04
    # print(modified_count)
    new_model = MultiChannelPAF(a2,a1,a0)
    return new_model

def change_all_HerPN_by_PAF_MutalChannel(model):
    # 获取模型的副本
    model_modules = list(model.named_modules())
    # 寻找对应为module
    for name, module in model_modules:
        if isinstance(module, HerPN2d):
            # 检查当前模块是否直接挂载在 model 上（即模块名字直接是 'relu1'、'relu2' 等）
            if name in model._modules:
                # 直接替换 model 中的属性
                # 替换的new_act
                new_act = change_HerPN2d_into_PAF_MutalChannel(module)
                setattr(model, name, new_act)  # 替换为新激活函数
            else:
                # 替换的new_act
                new_act = change_HerPN2d_into_PAF_MutalChannel(module)
                # 在layer,BLock的次级结构
                parent_name = name.rsplit('.', 1)[0]
                parent = dict(model.named_modules())[parent_name]  # 获取父模块
                # 删除父模块中的原 ReLU 层（如果存在）
                if hasattr(parent, name.rsplit('.', 1)[-1]):
                    delattr(parent, name.rsplit('.', 1)[-1])
                # 替换为新的 Chebyshev_Relu_MaxScale
                setattr(parent, name.rsplit('.', 1)[-1], new_act)  # 替换为新激活函数
    return model


def get_Aespa_MutalChannel_PAF_resnet20():
    model_path = './ResNet20_Aespa.pth'
    model = get_resnet20_HerPN(num_classes=10)
    model.load_state_dict(torch.load(model_path, map_location="cuda:0"), strict=False)
    model = change_all_HerPN_by_PAF_MutalChannel(model)
    return model

def get_Aespa_MutalChannel_PAF_resnet18():
    model_path = './ResNet18_Aespa.pth'
    model = get_resnet18_HerPN(num_classes=10)
    model.load_state_dict(torch.load(model_path, map_location="cuda:0"), strict=False)
    model = change_all_HerPN_by_PAF_MutalChannel(model)
    return model
