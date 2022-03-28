
from typing import Type, Any, Callable, Union, List, Optional
from model import BasicBlock, Bottleneck, _resnet, ResNet

def wide_resnet(pretrained: bool = False, m :int=6, width : int = 2, progress: bool = True, use_bottleneck: bool = False, **kwargs: Any) -> ResNet:
    r"""Wide ResNet-50-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_.
    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['width_per_group'] = 64 * width
    keys = ['n1', 'n2', 'n3', 'n4']
    kwargs ['cell_type'] = 'BasicBlock'
    print(kwargs)
    for key in keys:
        kwargs[key] = kwargs[key] * width
    print(kwargs)
    if use_bottleneck:
        return _resnet('wide_resnet', Bottleneck, [m, m, m],
                       pretrained, progress, **kwargs)
    else:    
        return _resnet('wide_resnet', BasicBlock, [m, m, m],
                       pretrained, progress, **kwargs)

