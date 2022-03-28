

from inspect import signature
from collections import namedtuple, OrderedDict
from typing import Type, Any, Callable, Union, List, Optional

import math 
import torch
from torch import nn
from torch import Tensor
from torch.nn import init
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from utils import SeparableConv2d, h_swish, _make_divisible, h_sigmoid, Hswish

from building_blocks import *
from global_layer import *

class ResNet(nn.Module):

    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        num_classes: int = 1000,
        zero_init_residual: bool = False,
        global_ft = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        n1: int = 64,
        n2: int = 128,
        n3: int = 128,
        n4: int = 128,
        cell_type : str = 'default',   
        args = None,
    ) -> None:
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        

        pde_args = {
                'K':             args.K, 
                'separable':     args.separable, 
                'nonlinear_pde': args.non_linear, 
                'cDx' :          args.cDx,
                'cDy' :          args.cDy,
                'dx' :           args.dx,
                'dy' :           args.dy,
                'dt' :           args.dt, 
                'init_h0_h':     args.init_h0_h,
                'use_silu' :     args.use_silu,
                'use_res' :      args.use_res,
                'constant_Dxy':  args.constant_Dxy,
                'custom_uv':     args.custom_uv,
                'custom_dxy':    args.custom_dxy,
                'no_f' :         args.no_f,
                'cell_type' :    cell_type,
                'old_style' :    False, # True, 
        }


        self.global_ft = global_ft

        self.inplanes = n1 #64 #16
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        
        self.separable = args.separable 
        self.layer1 = self._make_layer(block, n1, layers[0])
        self.layer2 = self._make_layer(block, n2, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, n3, layers[2], stride=2, dilate=replace_stride_with_dilation[1])

        self.original = len( layers ) == 3 
        if self.original == False:
            self.layer4 = self._make_layer(block, n4, layers[3], stride=2, dilate=replace_stride_with_dilation[1])
        else:
            assert ( n3 == n4 )
        
        if self.global_ft:
            self.global1 = GlobalFeatureBlock_Diffusion(n1, pde_args)   
            self.global2 = GlobalFeatureBlock_Diffusion(n2, pde_args) 
            self.global3 = GlobalFeatureBlock_Diffusion(n3, pde_args)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(n4 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                
                #nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                param = m.weight
                n = param.size(0) * param.size(2) * param.size(3)
                param.data.normal_().mul_(math.sqrt(2. / n))
                
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(self, block: Type[Union[BasicBlock, Bottleneck]], planes: int, blocks: int,
                    stride: int = 1, dilate: bool = False) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )
            #downsample = None

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer, separable=self.separable))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer, separable=self.separable))

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        
        debug = False
        if debug: print('x = ', x.size())
        
        x = self.conv1(x)
        if debug: print('conv1 = ', x.size())
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        if debug: print('layer1 = ', x.size())
            
        if self.global_ft:
            x = self.global1(x)
            if debug: print('global1 = ', x.size())
            
        x = self.layer2(x)
        if debug: print('layer2 = ', x.size())
            
        if self.global_ft:
            x = self.global2(x)
            if debug: print('global2 = ', x.size())
            
        x = self.layer3(x)
        if debug: print('layer3 = ', x.size())

        if self.global_ft:
            x = self.global3(x)
            if debug: print('global3 = ', x.size())
            
        if self.original == False:
            x = self.layer4(x)
            if debug: print('layer4 = ', x.size())

        x = self.avgpool(x)
        if debug: print('L4 avgpool = ', x.size())
            
        x = torch.flatten(x, 1)
        x = self.fc(x)
        if debug: print('fc = ', x.size())
        if debug: assert(1==2)

        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


def _resnet(
    arch: str,
    block: Type[Union[BasicBlock, Bottleneck]],
    layers: List[int],
    pretrained: bool,
    progress: bool,
    **kwargs: Any
) -> ResNet:
    model = ResNet(block, layers, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model


def resnet32(pretrained: bool = False, progress: bool = True, m : int = 5, **kwargs: Any) -> ResNet:
    return _resnet('resnet32', BasicBlock, [m, m, m, m], pretrained, progress, global_ft = False,
                   **kwargs)

def pdenet(pretrained: bool = False, progress: bool = True, m : int = 2, **kwargs: Any) -> ResNet:
    return _resnet('PDE32', BasicBlock, [m, m, m, m], pretrained, progress, global_ft = True,
                   **kwargs)


def resnet_original(pretrained: bool = False, progress: bool = True, m : int = 5, **kwargs: Any) -> ResNet:
    return _resnet('resnet-original', BasicBlock, [m, m, m], pretrained, progress, global_ft = False,
                   **kwargs)


def pdenet_original(pretrained: bool = False, progress: bool = True, m : int = 1, **kwargs: Any) -> ResNet:
    return _resnet('pde-original', BasicBlock, [m, m, m], pretrained, progress, global_ft = True,
                   **kwargs)




