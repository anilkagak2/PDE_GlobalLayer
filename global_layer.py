

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
from timm.models.efficientnet_blocks import InvertedResidual, DepthwiseSeparableConv
from timm.models.layers import create_conv2d, drop_path, make_divisible, create_act_layer

def get_init_block( planes, block_type = 'default', args = None ):
    assert(args is not None)

    separable = False 
    if args and 'separable' in args:
            separable = args['separable']

    if block_type == 'BasicBlock':
        init_h = BasicBlock(planes, planes, separable=separable)
    elif block_type == 'Bottleneck':
        init_h = Bottleneck(planes, planes, separable=separable, expansion=1)
    elif block_type == 'BasicDense':
        init_h = BasicDenseLayer( planes, bn_size=2 )  
    elif block_type == 'DwConv':
        dw_kernel_size = args.get('dw_kernel_size', 3) 
        init_h = nn.Conv2d(planes, planes, kernel_size=dw_kernel_size, stride=1, padding=1, groups=planes)
    elif block_type == 'FullConv':
        dw_kernel_size = args.get('dw_kernel_size', 3) 
        init_h = nn.Conv2d(planes, planes, kernel_size=dw_kernel_size, stride=1, padding=1)
    elif block_type == 'PwConv':
        init_h = nn.Conv2d(planes, planes, kernel_size=1, stride=1)
    elif block_type == 'DartCell':
        genotype, C_prev_prev, C_prev, C_curr, reduction, reduction_prev = args['genotype'], args['C_prev_prev'], args['C_prev'], args['C_curr'], args['reduction'], args['reduction_prev']
        init_h = Cell(genotype, C_prev_prev, C_prev, C_curr, reduction, reduction_prev)  
    elif block_type == 'InvertedResidualCell':

        in_chs, out_chs = args['in_chs'], args['out_chs']
        dw_kernel_size, exp_kernel_size, pw_kernel_size = args['dw_kernel_size'], args['exp_kernel_size'], args['pw_kernel_size'] 
        stride, dilation, pad_type = args['stride'], args['dilation'], args['pad_type']
        act_layer, noskip, exp_ratio = args['act_layer'], args['noskip'], args['exp_ratio']
        conv_kwargs = args.get('conv_kwargs', {}) 
        drop_path_rate = args['drop_path_rate']
        se_layer, norm_layer = args.get('se_layer', None), args['norm_layer']

        init_h = InvertedResidual( in_chs, out_chs, dw_kernel_size,
                 stride=stride, dilation=dilation, pad_type=pad_type, act_layer=act_layer, noskip=noskip,
                 exp_ratio=exp_ratio, exp_kernel_size=exp_kernel_size, pw_kernel_size=pw_kernel_size,
                 se_layer=se_layer, norm_layer=norm_layer, conv_kwargs=conv_kwargs, drop_path_rate=drop_path_rate)
    elif block_type == 'identity' or block_type == 'default':
        init_h = nn.Identity()
    else:
        print('Undefined cell type.', block_type)
        assert(1==2)

    return init_h    

class GlobalFeatureBlock_Diffusion(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        planes,
        args,
    ):
        super(GlobalFeatureBlock_Diffusion, self).__init__()

        norm_layer = args.get('norm_layer', None)
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        K = args.get('K', 10)
        nonlinear_pde = args.get('nonlinear_pde', True)
        separable = args.get('separable', True)
        pde_state = args.get('pde_state', 0)

        assert(nonlinear_pde == True)

        cDx = args.get('cDx', 1.)
        cDy = args.get('cDy', 1.)
        dx = args.get('dx', 1)
        dy = args.get('dy', 1)
        dt = args.get('dt', 0.2)

        init_h0_h = args.get('init_h0_h', False)
        self.use_f_for_g = args.get('use_f_for_g', False)
        use_diff_eps = args.get('use_diff_eps', True)
        use_silu = args.get('use_silu', False)
        use_res = args.get('use_res', False)
        use_cDs = args.get('use_cDs', False)
        use_dw = args.get('use_dw', False)
        use_dot = args.get('use_dot', False)
        drop_path_rate = args.get('drop_path_rate', 0.)
        constant_Dxy = args.get('constant_Dxy', False)
        no_f = args.get('no_f', False)
        block_type = args.get('cell_type', 'default')

        dw_kernel_size = args.get('dw_kernel_size', 3) 
        pw_kernel_size = args.get('pw_kernel_size', 1) 
        exp_kernel_size = args.get('exp_kernel_size', 1) 
        se_layer = args.get('se_layer', None) 
        old_style = args.get('old_style', False) 

        dilation = args.get('dilation', 1)
        pad_type = args.get('pad_type', '')
        stride = args.get('stride', 1)
        in_chs = args.get('in_chs', planes)
        out_chs = args.get('out_chs', planes)
        if 'out_chs' in args:
            planes = out_chs

        print('Global Feature Block Diffusion : (K, planes, nonlinear_pde, pde_state, block_type)', K, planes, nonlinear_pde, pde_state, block_type)
        print(' c-Dxy, dt, no_f, use_silu, use_res, cDx, cDy, init_h0_h, dx, dy, use_dot, use_cDs, drop_path_rate ', constant_Dxy, dt, no_f, use_silu, use_res, cDx, cDy, init_h0_h, dx, dy, use_dot, use_cDs, drop_path_rate)

        if block_type == 'DartCell':
            planes = args['C_prev'] #C_curr
            print('[GB] planes, C_prev_prev, C_prev, C_curr = ', planes, args['C_prev_prev'], args['C_prev'], args['C_curr'])

        self.pde_state = pde_state
        self.nonlinear_pde = nonlinear_pde
        self.K = K
        
        self.relu = nn.ReLU(inplace=True)
        if use_silu:
            self.act = nn.SiLU(inplace=True) 
        else:
            self.act = nn.ReLU(inplace=True)
        if 'act_layer' in args:
            self.act = args['act_layer'](inplace=True)

        self.init_h = get_init_block( planes, block_type, args )
        self.bn_out = norm_layer(planes)

        self.block_type = block_type
        self.init_h0_h = init_h0_h
        self.dx = dx
        self.dy = dy
        self.cDx = cDx
        self.cDy = cDy
        self.use_res = use_res
        self.use_dot = use_dot
        self.no_f = no_f
        self.dt = dt
        self.constant_Dxy = constant_Dxy

        self.drop_path_rate = drop_path_rate
        self.stride = stride
        self.in_chs = in_chs
        self.out_chs = out_chs
        self.planes = planes

        # TODO Ideally I should allow linear counter-part. But this needs some work
        if self.nonlinear_pde:
            if args['custom_uv'] == '':
              if use_dw:
                self.convg = DepthwiseSeparableConv(planes, planes, dw_kernel_size, stride=1, dilation=dilation, pad_type=pad_type, 
                            act_layer=act_layer, noskip=True, pw_kernel_size=pw_kernel_size, se_layer=se_layer, norm_layer=norm_layer, drop_path_rate=drop_path_rate )
                self.convg1 = DepthwiseSeparableConv(planes, planes, dw_kernel_size, stride=1, dilation=dilation, pad_type=pad_type, 
                            act_layer=act_layer, noskip=True, pw_kernel_size=pw_kernel_size, se_layer=se_layer, norm_layer=norm_layer, drop_path_rate=drop_path_rate )
              else:
                if old_style:
                    self.convg  = nn.Conv2d(planes, planes, kernel_size=dw_kernel_size, stride=1, padding=1, groups=planes)
                    self.convg1 = nn.Conv2d(planes, planes, kernel_size=dw_kernel_size, stride=1, padding=1, groups=planes)
                else:    
                    self.convg  = create_conv2d(planes, planes, kernel_size=dw_kernel_size, stride=1, dilation=dilation, padding=pad_type, depthwise=True)
                    self.convg1 = create_conv2d(planes, planes, kernel_size=dw_kernel_size, stride=1, dilation=dilation, padding=pad_type, depthwise=True)
            else: 
              print('Custom uv ', args['custom_uv'])
              self.convg  = get_init_block( planes, block_type = args['custom_uv'], args = args )
              self.convg1 = get_init_block( planes, block_type = args['custom_uv'], args = args )

            if use_diff_eps:
                self.bng = norm_layer(planes, planes)
                self.bng1 = norm_layer(planes, planes)
            else:
                self.bng = norm_layer(planes)
                self.bng1 = norm_layer(planes)

            if constant_Dxy == False:
              if args['custom_dxy'] == '':
                if use_cDs == False:
                    if old_style:
                        self.convDx = nn.Conv2d(planes, planes, kernel_size=dw_kernel_size, stride=1, padding=1, groups=planes)
                        self.convDy = nn.Conv2d(planes, planes, kernel_size=dw_kernel_size, stride=1, padding=1, groups=planes)
                    else:    
                        self.convDx = create_conv2d(planes, planes, kernel_size=dw_kernel_size, stride=1, dilation=dilation, padding=pad_type, depthwise=True)
                        self.convDy = create_conv2d(planes, planes, kernel_size=dw_kernel_size, stride=1, dilation=dilation, padding=pad_type, depthwise=True)
                else:
                    self.convDx = DepthwiseSeparableConv(planes, planes, dw_kernel_size, stride=1, dilation=dilation, pad_type=pad_type, 
                            act_layer=act_layer, noskip=True, pw_kernel_size=pw_kernel_size, se_layer=se_layer, norm_layer=norm_layer, drop_path_rate=drop_path_rate )
                    self.convDy = DepthwiseSeparableConv(planes, planes, dw_kernel_size, stride=1, dilation=dilation, pad_type=pad_type, 
                            act_layer=act_layer, noskip=True, pw_kernel_size=pw_kernel_size, se_layer=se_layer, norm_layer=norm_layer, drop_path_rate=drop_path_rate )
              else: 
                print('Custom xy ', args['custom_dxy'])
                self.convDx = get_init_block( planes, block_type = args['custom_dxy'], args = args )
                self.convDy = get_init_block( planes, block_type = args['custom_dxy'], args = args )

              if use_diff_eps:
                    self.bnDx = norm_layer(planes, planes)
                    self.bnDy = norm_layer(planes, planes)
              else:    
                    self.bnDx = norm_layer(planes)
                    self.bnDy = norm_layer(planes)

    def feature_info(self, location):
        info = dict(module='', hook_type='', num_chs=self.planes)
        return info


    def forward(self, s0, s1 = None, drop_path=None):
        if self.block_type == 'DartCell':
            f = s1
            h = self.init_h( s0, s1, drop_path )
            #h = self.init_h( s1, f, drop_path )
        else:
            f = s0
            h = self.init_h( f )

        if (self.stride != 1) or (self.in_chs != self.out_chs):  
            f = h
        residual = f

        debug = False
        if debug: print('f = ', f.size())
        if debug: print('h = ', h.size())
        
        if self.init_h0_h :
            h0 = h 
        else:
            h0 = f  
   
        g0 = h 
        if self.use_f_for_g: g0 = f 
        if self.use_dot:
            g = self.act( self.bng(self.convg(g0)) * self.act( self.bng1(self.convg1(g0)) ) )
            g1 = g
        else:  
            g = self.act( self.bng(self.convg(g0)) )
            g1 = self.act( self.bng1(self.convg1(g0)) )
            
        dt = self.dt 
        dx = self.dx 
        dy = self.dy 

        if self.constant_Dxy:
            Dx = self.cDx 
            Dy = self.cDy 
        else:
            if self.use_dot:
                Dx  = self.act( self.bnDx(self.convDx(h)) * self.act( self.bnDy(self.convDy(h)) ) )
                Dy  = Dx 
            else:
                Dx  = self.act( self.bnDx(self.convDx(h)) )
                Dy  = self.act( self.bnDy(self.convDy(h)) )

        ux = (1. / (2*dx)) * ( torch.roll(g, dx, dims=2)  - torch.roll(g, -dx, dims=2) )
        vy = (1. / (2*dy)) * ( torch.roll(g1, dy, dims=3) - torch.roll(g1, -dy, dims=3) )

        Ax = g  * (dt / dx)
        Ay = g1 * (dt / dy)
        Bx = Dx * (dt / (dx*dx))
        By = Dy * (dt / (dy*dy))
        E  = (ux + vy) * dt

        D = (1. / (1 + 2*Bx + 2*By))

        for k in range(self.K):
            if debug: print('f = ', f.size())

            prev_h = h
                
            h = D  *   (   (1 - 2*Bx - 2*By) * h0 - 2 * E * h 
                         + (-Ax  + 2*Bx) * torch.roll(h, dx, dims=2) 
                         + ( Ax  + 2*Bx) * torch.roll(h, -dx, dims=2) 
                         + (-Ay  + 2*By) * torch.roll(h, dy, dims=3)  
                         + ( Ay  + 2*By) * torch.roll(h, -dy, dims=3)  
                        ) # + 2 * dt * f )
            if self.no_f == False:
                h = h + D * 2 * dt * f 

            h0 = prev_h
                
        
        h = self.bn_out(h)
        h = self.act(h)
        if self.use_res: 
            if self.drop_path_rate > 0.:
                h = drop_path(h, self.drop_path_rate, self.training)
            h = h + residual #s1

        if debug: print('out h = ', h.size())
        return h

