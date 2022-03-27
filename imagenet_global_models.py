""" 
Global Layered variants of the known family of Imagenet Models 
Uses codebase from timm repository

"""
from functools import partial
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
from timm.models.efficientnet_blocks import SqueezeExcite
from efficientnet_builder import EfficientNetBuilder, decode_arch_def, efficientnet_init_weights, round_channels, resolve_bn_args, resolve_act_layer, BN_EPS_TF_DEFAULT
from timm.models.features import FeatureInfo, FeatureHooks
from timm.models.helpers import build_model_with_cfg, default_cfg_for_features
from timm.models.layers import create_conv2d, create_classifier,  SelectAdaptivePool2d, Linear, get_act_fn, hard_sigmoid
from timm.models.registry import register_model

#from timm.models.mobilenetv3 import MobileNetV3
#from timm.models.efficientnet import EfficientNet

def _cfg(url='', **kwargs):
    return {
        'url': url, 'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': (7, 7),
        'crop_pct': 0.875, 'interpolation': 'bicubic',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'conv_stem', 'classifier': 'classifier',
        **kwargs
    }


default_cfgs = {

    'new_mobilenetv2_100': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/mobilenetv2_100_ra-b33bc2c4.pth'),

    'new_efficientnet_b0': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/efficientnet_b0_ra-3dd342df.pth'),

    'new_mobilenetv3_large_100': _cfg(
        pool_size=(1,1),  
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/mobilenetv3_large_100_ra-f55367f5.pth'),

    'global_mobilenetv2_100': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/mobilenetv2_100_ra-b33bc2c4.pth'),

    'global_efficientnet_b0': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/efficientnet_b0_ra-3dd342df.pth'),

    'global_mobilenetv3_large_100': _cfg(
        pool_size=(1,1),  
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/mobilenetv3_large_100_ra-f55367f5.pth'),

}


class MobileNetV3(nn.Module):
    def __init__(
            self, block_args, num_classes=1000, in_chans=3, stem_size=16, fix_stem=False, num_features=1280,
            head_bias=True, pad_type='', act_layer=None, norm_layer=None, se_layer=None, se_from_exp=True,
            round_chs_fn=round_channels, drop_rate=0., drop_path_rate=0., global_pool='avg'):
        super(MobileNetV3, self).__init__()
        act_layer = act_layer or nn.ReLU
        norm_layer = norm_layer or nn.BatchNorm2d
        se_layer = se_layer or SqueezeExcite
        self.num_classes = num_classes
        self.num_features = num_features
        self.drop_rate = drop_rate

        # Stem
        if not fix_stem:
            stem_size = round_chs_fn(stem_size)
        self.conv_stem = create_conv2d(in_chans, stem_size, 3, stride=2, padding=pad_type)
        self.bn1 = norm_layer(stem_size)
        self.act1 = act_layer(inplace=True)

        # Middle stages (IR/ER/DS Blocks)
        builder = EfficientNetBuilder(
            output_stride=32, pad_type=pad_type, round_chs_fn=round_chs_fn, se_from_exp=se_from_exp,
            act_layer=act_layer, norm_layer=norm_layer, se_layer=se_layer, drop_path_rate=drop_path_rate)
        self.blocks = nn.Sequential(*builder(stem_size, block_args))
        self.feature_info = builder.features
        head_chs = builder.in_chs

        # Head + Pooling
        self.global_pool = SelectAdaptivePool2d(pool_type=global_pool)
        num_pooled_chs = head_chs * self.global_pool.feat_mult()
        self.conv_head = create_conv2d(num_pooled_chs, self.num_features, 1, padding=pad_type, bias=head_bias)
        self.act2 = act_layer(inplace=True)
        self.flatten = nn.Flatten(1) if global_pool else nn.Identity()  # don't flatten if pooling disabled
        self.classifier = Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

        efficientnet_init_weights(self)

    def as_sequential(self):
        layers = [self.conv_stem, self.bn1, self.act1]
        layers.extend(self.blocks)
        layers.extend([self.global_pool, self.conv_head, self.act2])
        layers.extend([nn.Flatten(), nn.Dropout(self.drop_rate), self.classifier])
        return nn.Sequential(*layers)

    def get_classifier(self):
        return self.classifier

    def reset_classifier(self, num_classes, global_pool='avg'):
        self.num_classes = num_classes
        # cannot meaningfully change pooling of efficient head after creation
        self.global_pool = SelectAdaptivePool2d(pool_type=global_pool)
        self.flatten = nn.Flatten(1) if global_pool else nn.Identity()  # don't flatten if pooling disabled
        self.classifier = Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        x = self.conv_stem(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.blocks(x)
        x = self.global_pool(x)
        x = self.conv_head(x)
        x = self.act2(x)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.flatten(x)
        if self.drop_rate > 0.:
            x = F.dropout(x, p=self.drop_rate, training=self.training)
        return self.classifier(x)




class EfficientNet(nn.Module):
    def __init__(self, block_args, num_classes=1000, num_features=1280, in_chans=3, stem_size=32, fix_stem=False,
                 output_stride=32, pad_type='', round_chs_fn=round_channels, act_layer=None, norm_layer=None,
                 se_layer=None, drop_rate=0., drop_path_rate=0., global_pool='avg'):
        super(EfficientNet, self).__init__()
        act_layer = act_layer or nn.ReLU
        norm_layer = norm_layer or nn.BatchNorm2d
        se_layer = se_layer or SqueezeExcite
        self.num_classes = num_classes
        self.num_features = num_features
        self.drop_rate = drop_rate

        # Stem
        if not fix_stem:
            stem_size = round_chs_fn(stem_size)
        self.conv_stem = create_conv2d(in_chans, stem_size, 3, stride=2, padding=pad_type)
        self.bn1 = norm_layer(stem_size)
        self.act1 = act_layer(inplace=True)

        # Middle stages (IR/ER/DS Blocks)
        builder = EfficientNetBuilder(
            output_stride=output_stride, pad_type=pad_type, round_chs_fn=round_chs_fn,
            act_layer=act_layer, norm_layer=norm_layer, se_layer=se_layer, drop_path_rate=drop_path_rate)
        self.blocks = nn.Sequential(*builder(stem_size, block_args))
        self.feature_info = builder.features
        head_chs = builder.in_chs

        # Head + Pooling
        self.conv_head = create_conv2d(head_chs, self.num_features, 1, padding=pad_type)
        self.bn2 = norm_layer(self.num_features)
        self.act2 = act_layer(inplace=True)
        self.global_pool, self.classifier = create_classifier(
            self.num_features, self.num_classes, pool_type=global_pool)

        efficientnet_init_weights(self)

    def as_sequential(self):
        layers = [self.conv_stem, self.bn1, self.act1]
        layers.extend(self.blocks)
        layers.extend([self.conv_head, self.bn2, self.act2, self.global_pool])
        layers.extend([nn.Dropout(self.drop_rate), self.classifier])
        return nn.Sequential(*layers)

    def get_classifier(self):
        return self.classifier

    def reset_classifier(self, num_classes, global_pool='avg'):
        self.num_classes = num_classes
        self.global_pool, self.classifier = create_classifier(
            self.num_features, self.num_classes, pool_type=global_pool)

    def forward_features(self, x):
        x = self.conv_stem(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.blocks(x)
        x = self.conv_head(x)
        x = self.bn2(x)
        x = self.act2(x)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.global_pool(x)
        if self.drop_rate > 0.:
            x = F.dropout(x, p=self.drop_rate, training=self.training)
        return self.classifier(x)



def _create_effnet(variant, pretrained=False, **kwargs):
    features_only = False
    model_cls = EfficientNet
    kwargs_filter = None
    if kwargs.pop('features_only', False):
        assert(1==2)
    model = build_model_with_cfg(
        model_cls, variant, pretrained,
        default_cfg=default_cfgs[variant],
        pretrained_strict=not features_only,
        kwargs_filter=kwargs_filter,
        **kwargs)
    return model

def _create_mnv3(variant, pretrained=False, **kwargs):
    features_only = False
    model_cls = MobileNetV3
    kwargs_filter = None
    if kwargs.pop('features_only', False):
        assert(1==2)
    model = build_model_with_cfg(
        model_cls, variant, pretrained,
        default_cfg=default_cfgs[variant],
        pretrained_strict=not features_only,
        kwargs_filter=kwargs_filter,
        **kwargs)
    return model


def _gen_mobilenet_v2(
        variant, channel_multiplier=1.0, depth_multiplier=1.0, fix_stem_head=False, pretrained=False, **kwargs):
    """ Generate MobileNet-V2 network
    Ref impl: https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet_v2.py
    Paper: https://arxiv.org/abs/1801.04381
    """
    if 'global' in variant:
        arch_def = [
            ['ds_r1_k3_s1_c16'], 
            ['ir_r1_k3_s2_e6_c24', 'gr_r1_k3_s1_e6_c24_nre_j1_rs1_xy1' ],
            ['ir_r1_k3_s2_e6_c32', 'gr_r1_k3_s1_e6_c32_nre_j1_rs1_xy1' ],
            ['ir_r1_k3_s2_e6_c64', 'gr_r1_k3_s1_e6_c64_nre_j1_rs1_xy1' ],
            ['ir_r1_k3_s1_e6_c96', 'gr_r1_k3_s1_e6_c96_j1_rs1_xy1' ],
            ['ir_r1_k3_s2_e6_c160', 'gr_r1_k3_s1_e6_c160_j1_rs1_xy1' ],
            ['ir_r1_k3_s1_e6_c320'],
        ]
    else:
        arch_def = [
            ['ds_r1_k3_s1_c16'],
            ['ir_r2_k3_s2_e6_c24'],
            ['ir_r3_k3_s2_e6_c32'],
            ['ir_r4_k3_s2_e6_c64'],
            ['ir_r3_k3_s1_e6_c96'],
            ['ir_r3_k3_s2_e6_c160'],
            ['ir_r1_k3_s1_e6_c320'],
        ]
    round_chs_fn = partial(round_channels, multiplier=channel_multiplier)
    model_kwargs = dict(
        block_args=decode_arch_def(arch_def, depth_multiplier=depth_multiplier, fix_first_last=fix_stem_head),
        num_features=1280 if fix_stem_head else max(1280, round_chs_fn(1280)),
        stem_size=32,
        fix_stem=fix_stem_head,
        round_chs_fn=round_chs_fn,
        norm_layer=kwargs.pop('norm_layer', None) or partial(nn.BatchNorm2d, **resolve_bn_args(kwargs)),
        act_layer=resolve_act_layer(kwargs, 'relu6'),
        **kwargs
    )
    model = _create_effnet(variant, pretrained, **model_kwargs)
    return model


def _gen_efficientnet(variant, channel_multiplier=1.0, depth_multiplier=1.0, pretrained=False, **kwargs):
    """Creates an EfficientNet model.
    Ref impl: https://github.com/tensorflow/tpu/blob/master/models/official/efficientnet/efficientnet_model.py
    Paper: https://arxiv.org/abs/1905.11946
    """
    if 'global' in variant:
        arch_def = [
            ['ds_r1_k3_s1_e1_c16_se0.25'], 
            ['ir_r1_k3_s2_e6_c24_se0.25',  'gr_r1_k3_s1_e6_c24_se0.25_j1_xy1_rs1_udot1_dx1_dy1_dt0.1_cDx2._cDy2.'],
            ['ir_r1_k5_s2_e6_c40_se0.25',  'gr_r1_k5_s1_e6_c40_se0.25_j1_xy1_rs1_udot1_dx2_dy2_dt0.1_cDx2._cDy2.'],
            ['ir_r1_k3_s2_e6_c80_se0.25',  'gr_r1_k3_s1_e6_c80_se0.25_j1_xy1_rs1_udot1_ucds1'],
            ['ir_r1_k5_s1_e6_c112_se0.25', 'gr_r1_k5_s1_e6_c112_se0.25_j1_xy1_rs1_udot1_ucds1'],
            ['ir_r1_k5_s2_e6_c192_se0.25', 'gr_r1_k5_s1_e6_c192_se0.25_j1_xy1_rs1_udot1'],
            ['ir_r1_k3_s1_e6_c320_se0.25'],
        ]
    else:
        arch_def = [
            ['ds_r1_k3_s1_e1_c16_se0.25'],
            ['ir_r2_k3_s2_e6_c24_se0.25'],
            ['ir_r2_k5_s2_e6_c40_se0.25'],
            ['ir_r3_k3_s2_e6_c80_se0.25'],
            ['ir_r3_k5_s1_e6_c112_se0.25'],
            ['ir_r4_k5_s2_e6_c192_se0.25'],
            ['ir_r1_k3_s1_e6_c320_se0.25'],
        ]
    round_chs_fn = partial(round_channels, multiplier=channel_multiplier)
    model_kwargs = dict(
        block_args=decode_arch_def(arch_def, depth_multiplier),
        num_features=round_chs_fn(1280),
        stem_size=32,
        round_chs_fn=round_chs_fn,
        act_layer=resolve_act_layer(kwargs, 'swish'),
        norm_layer=kwargs.pop('norm_layer', None) or partial(nn.BatchNorm2d, **resolve_bn_args(kwargs)),
        **kwargs,
    )
    model = _create_effnet(variant, pretrained, **model_kwargs)
    return model


def _gen_mobilenet_v3(variant, channel_multiplier=1.0, pretrained=False, **kwargs):
    """Creates a MobileNet-V3 model.
    Ref impl: ?
    Paper: https://arxiv.org/abs/1905.02244
    """
    if 'small' in variant:
        assert(1==2)
    else:
        num_features = 1280
        if 'minimal' in variant:
            assert(1==2)
        else:
            act_layer = resolve_act_layer(kwargs, 'hard_swish')
            if 'global' in variant:
                arch_def = [
                    ['ds_r1_k3_s1_e1_c16_nre'],   # relu
                    ['ir_r1_k3_s2_e4_c24_nre', 'gr_r1_k3_s1_e3_c24_nre_j1_xy1_rs1'],  # relu
                    ['ir_r1_k5_s2_e3_c40_se0.25_nre', 'gr_r1_k5_s1_e3_c40_se0.25_nre_j1_rs1_xy1_udot1_dx1_dy1_dt0.2_cDx1._cDy1._ucds1'],  # relu
                    ['ir_r1_k3_s2_e6_c80', 'gr_r1_k3_s1_e6_c80_j1_xy1_udot1_ucds1_rs1'],  # hard-swish
                    ['ir_r1_k3_s1_e6_c112_se0.25', 'gr_r1_k3_s1_e6_c112_se0.25_j1_xy1_rs1_udot1'],  # hard-swish
                    ['gr_r1_k5_s2_e6_c160_se0.25_j1_xy1_rs1_udot1'],  # hard-swish
                    ['cn_r1_k1_s1_c960'],  # hard-swish
                ]
            else:
                arch_def = [
                    ['ds_r1_k3_s1_e1_c16_nre'],  # relu
                    ['ir_r1_k3_s2_e4_c24_nre', 'ir_r1_k3_s1_e3_c24_nre'],  # relu
                    ['ir_r3_k5_s2_e3_c40_se0.25_nre'],  # relu
                    ['ir_r1_k3_s2_e6_c80', 'ir_r1_k3_s1_e2.5_c80', 'ir_r2_k3_s1_e2.3_c80'],  # hard-swish
                    ['ir_r2_k3_s1_e6_c112_se0.25'],  # hard-swish
                    ['ir_r3_k5_s2_e6_c160_se0.25'],  # hard-swish
                    ['cn_r1_k1_s1_c960'],  # hard-swish
            ]

    se_layer = partial(SqueezeExcite, gate_layer='hard_sigmoid', force_act_layer=nn.ReLU, rd_round_fn=round_channels)
    model_kwargs = dict(
        block_args=decode_arch_def(arch_def),
        num_features=num_features,
        stem_size=16,
        fix_stem=channel_multiplier < 0.75,
        round_chs_fn=partial(round_channels, multiplier=channel_multiplier),
        norm_layer=partial(nn.BatchNorm2d, **resolve_bn_args(kwargs)),
        act_layer=act_layer,
        se_layer=se_layer,
        **kwargs,
    )
    model = _create_mnv3(variant, pretrained, **model_kwargs)
    return model


def new_mobilenetv2_100(pretrained=False, **kwargs):
    """ MobileNet V2 w/ 1.0 channel multiplier """
    model = _gen_mobilenet_v2('new_mobilenetv2_100', 1.0, pretrained=pretrained, **kwargs)
    return model

def global_mobilenetv2_100(pretrained=False, **kwargs):
    """ MobileNet V2 w/ 1.0 channel multiplier """
    model = _gen_mobilenet_v2('global_mobilenetv2_100', 1.0, pretrained=pretrained, **kwargs)
    return model



def new_mobilenetv3_large_100(pretrained=False, **kwargs):
    """ MobileNet V3 """
    model = _gen_mobilenet_v3('new_mobilenetv3_large_100', 1.0, pretrained=pretrained, **kwargs)
    return model

def global_mobilenetv3_large_100(pretrained=False, **kwargs):
    """ MobileNet V3 """
    model = _gen_mobilenet_v3('global_mobilenetv3_large_100', 1.0, pretrained=pretrained, **kwargs)
    return model



def new_efficientnet_b0(pretrained=False, **kwargs):
    """ EfficientNet-B0 """
    # NOTE for train, drop_rate should be 0.2, drop_path_rate should be 0.2
    model = _gen_efficientnet(
        'new_efficientnet_b0', channel_multiplier=1.0, depth_multiplier=1.0, pretrained=pretrained, **kwargs)
    return model


def global_efficientnet_b0(pretrained=False, **kwargs):
    """ EfficientNet-B0 """
    # NOTE for train, drop_rate should be 0.2, drop_path_rate should be 0.2
    model = _gen_efficientnet(
        'global_efficientnet_b0', channel_multiplier=1.0, depth_multiplier=1.0, pretrained=pretrained, **kwargs)
    return model


def get_global_model( model_name, pretrained=False, **kwargs ):
    print(' --- model_name  ', model_name)
    if model_name == 'new_mobilenetv2_100':
        model = new_mobilenetv2_100(pretrained=pretrained, **kwargs)
    elif model_name == 'global_mobilenetv2_100':
        model = global_mobilenetv2_100(pretrained=pretrained, **kwargs)
    elif model_name == 'new_mobilenetv3_large_100':
        model = new_mobilenetv3_large_100(pretrained=pretrained, **kwargs)
    elif model_name == 'global_mobilenetv3_large_100':
        model = global_mobilenetv3_large_100(pretrained=pretrained, **kwargs)
    elif model_name == 'new_efficientnet_b0':
        model = new_efficientnet_b0(pretrained=pretrained, **kwargs)
    elif model_name == 'global_efficientnet_b0':
        model = global_efficientnet_b0(pretrained=pretrained, **kwargs)
    else:    
        assert(1==2)
    return model    

