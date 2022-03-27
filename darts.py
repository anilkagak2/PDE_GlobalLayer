
import torch
import torch.nn as nn
from torch.autograd import Variable
from utils import drop_path
import torch.nn.functional as F

from building_blocks import *
from global_layer import *



class AuxiliaryHeadCIFAR(nn.Module):

  def __init__(self, C, num_classes):
    """assuming input size 8x8"""
    super(AuxiliaryHeadCIFAR, self).__init__()
    self.features = nn.Sequential(
      nn.ReLU(inplace=True),
      nn.AvgPool2d(5, stride=3, padding=0, count_include_pad=False), # image size = 2 x 2
      nn.Conv2d(C, 128, 1, bias=False),
      nn.BatchNorm2d(128),
      nn.ReLU(inplace=True),
      nn.Conv2d(128, 768, 2, bias=False),
      nn.BatchNorm2d(768),
      nn.ReLU(inplace=True)
    )
    self.classifier = nn.Linear(768, num_classes)

  def forward(self, x):
    x = self.features(x)
    x = self.classifier(x.view(x.size(0),-1))
    return x

class DARTS_NetworkCIFAR(nn.Module):

  def __init__(self, C, num_classes, layers, auxiliary, genotype, args=None, global_ft=False):
    super(DARTS_NetworkCIFAR, self).__init__()

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
                'constant_Dxy':  args.constant_Dxy,
                'no_f' :         args.no_f,
                'cell_type' :   'DartCell',
                'old_style' :   False, # True, 
        }

    self._layers = layers
    self._auxiliary = auxiliary
    self.drop_path_prob = 0.001

    stem_multiplier = 3
    C_curr = stem_multiplier*C
    self.stem = nn.Sequential(
      nn.Conv2d(3, C_curr, 3, padding=1, bias=False),
      nn.BatchNorm2d(C_curr)
    )
    
    C_prev_prev, C_prev, C_curr = C_curr, C_curr, C
    self.cells = nn.ModuleList()
    reduction_prev = False
    self._idx_ = 0
    if global_ft: 
        self._idx_ = 2
    for i in range(layers):
      if i in [layers//3, 2*layers//3]:
        if global_ft:  
          # Add global block
          pde_args['genotype'], pde_args['C_prev_prev'], pde_args['C_prev'], pde_args['C_curr'], pde_args['reduction'], pde_args['reduction_prev'] = genotype, C_prev_prev, C_prev, C_curr, reduction, reduction_prev
          cell = GlobalFeatureBlock_Diffusion( C_prev, pde_args )

          self.cells += [cell]
          C_prev_prev, C_prev = C_prev, cell.init_h.multiplier*C_curr
          reduction_prev = reduction
          #print('C_prev_prev, C_prev, C_curr = ', C_prev_prev, C_prev, C_curr)

        C_curr *= 2
        reduction = True
      else:
        reduction = False
      cell = Cell(genotype, C_prev_prev, C_prev, C_curr, reduction, reduction_prev)
      reduction_prev = reduction
      self.cells += [cell]
      C_prev_prev, C_prev = C_prev, cell.multiplier*C_curr
      if i == 2*layers//3:
        C_to_auxiliary = C_prev

    if auxiliary:
      self.auxiliary_head = AuxiliaryHeadCIFAR(C_to_auxiliary, num_classes)
    self.global_pooling = nn.AdaptiveAvgPool2d(1)
    self.classifier = nn.Linear(C_prev, num_classes)

  def forward(self, input):
    logits_aux = None
    s0 = s1 = self.stem(input)
    for i, cell in enumerate(self.cells):
      s0, s1 = s1, cell(s0, s1, self.drop_path_prob)
      if i == self._idx_ + 2*self._layers//3:
        if self._auxiliary and self.training:
          logits_aux = self.auxiliary_head(s1)
    out = self.global_pooling(s1)
    logits = self.classifier(out.view(out.size(0),-1))
    return logits, logits_aux



