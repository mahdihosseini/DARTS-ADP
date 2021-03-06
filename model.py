import torch
import torch.nn as nn
from operations import *
from torch.autograd import Variable
from utils import drop_path
from collections import namedtuple

Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')

class Cell(nn.Module):

  def __init__(self, genotype, C_prev_prev, C_prev, C, reduction, reduction_prev):
    super(Cell, self).__init__()
    print(C_prev_prev, C_prev, C)

    if reduction_prev:
      self.preprocess0 = FactorizedReduce(C_prev_prev, C)
    else:
      self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0)
    self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0)
    
    if reduction:
      op_names, indices = zip(*genotype.reduce)
      concat = genotype.reduce_concat
    else:
      op_names, indices = zip(*genotype.normal)
      concat = genotype.normal_concat
    self._compile(C, op_names, indices, concat, reduction)

  def _compile(self, C, op_names, indices, concat, reduction):
    assert len(op_names) == len(indices)
    self._steps = len(op_names) // 2
    self._concat = concat
    self.multiplier = len(concat)

    self._ops = nn.ModuleList()
    for name, index in zip(op_names, indices):
      stride = 2 if reduction and index < 2 else 1
      op = OPS[name](C, stride, True)
      self._ops += [op]
    self._indices = indices

  def forward(self, s0, s1, drop_prob):
    s0 = self.preprocess0(s0)
    s1 = self.preprocess1(s1)

    states = [s0, s1]
    for i in range(self._steps):
      h1 = states[self._indices[2*i]]
      h2 = states[self._indices[2*i+1]]
      op1 = self._ops[2*i]
      op2 = self._ops[2*i+1]
      h1 = op1(h1)
      h2 = op2(h2)
      if self.training and drop_prob > 0.:
        if not isinstance(op1, Identity):
          h1 = drop_path(h1, drop_prob)
        if not isinstance(op2, Identity):
          h2 = drop_path(h2, drop_prob)
      s = h1 + h2
      states += [s]
    return torch.cat([states[i] for i in self._concat], dim=1)

class AuxiliaryHeadADP(nn.Module):

  def __init__(self, C, num_classes):
    """assuming input size 17x17"""
    super(AuxiliaryHeadADP, self).__init__()
    self.features = nn.Sequential(
      nn.ReLU(inplace=True),
      nn.AdaptiveAvgPool2d(2), 
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


class DARTS_ADP(nn.Module):

  def __init__(self, C, num_classes, layers, auxiliary, genotype):
    super(DARTS_ADP, self).__init__()
    self._layers = layers
    self._auxiliary = auxiliary

    self.stem = nn.Sequential(
      nn.Conv2d(3, C // 2, kernel_size=3, stride=2, padding=1, bias=False),
      nn.BatchNorm2d(C // 2),
      nn.ReLU(inplace=True),
      nn.Conv2d(C // 2, C, 3, stride=2, padding=1, bias=False),
      nn.BatchNorm2d(C),
    )
    
    C_prev_prev, C_prev, C_curr = C, C, C
    self.cells = nn.ModuleList()
    reduction_prev = False
    for i in range(layers):
      if i in [layers//3, 2*layers//3]:
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
      self.auxiliary_head = AuxiliaryHeadADP(C_to_auxiliary, num_classes)
    self.global_pooling = nn.AdaptiveAvgPool2d(1)
    self.classifier = nn.Linear(C_prev, num_classes)

  def forward(self, input):
    logits_aux = None
    s0 = s1 = self.stem(input)
    for i, cell in enumerate(self.cells):
      s0, s1 = s1, cell(s0, s1, self.drop_path_prob)
      if i == 2*self._layers//3:
        if self._auxiliary and self.training:
          logits_aux = self.auxiliary_head(s1)
    out = self.global_pooling(s1)
    logits = self.classifier(out.view(out.size(0),-1))
    return logits, logits_aux

def DARTS_ADP_N2(num_classes, auxiliary=False):

  genotype = Genotype(
    normal=[
      ('max_pool_3x3', 1), 
      ('max_pool_3x3', 0), 
      ('dil_conv_5x5', 2), 
      ('max_pool_3x3', 1)
    ], 
    normal_concat=range(2, 4), 
    reduce=[
      ('sep_conv_5x5', 0), 
      ('max_pool_3x3', 1), 
      ('max_pool_3x3', 2), 
      ('dil_conv_5x5', 0)
    ], 
    reduce_concat=range(2, 4)
  )

  return DARTS_ADP(36, num_classes, 4, auxiliary, genotype)

def DARTS_ADP_N3(num_classes, auxiliary=False):

  genotype = Genotype(
    normal=[
      ('max_pool_3x3', 0), 
      ('max_pool_3x3', 1), 
      ('sep_conv_5x5', 2), 
      ('max_pool_3x3', 1), 
      ('dil_conv_5x5', 3), 
      ('max_pool_3x3', 1)
    ], 
    normal_concat=range(2, 5), 
    reduce=[
      ('max_pool_3x3', 0), 
      ('dil_conv_5x5', 1), 
      ('max_pool_3x3', 0), 
      ('max_pool_3x3', 2), 
      ('skip_connect', 1), 
      ('max_pool_3x3', 0)
    ], 
    reduce_concat=range(2, 5)
  )

  return DARTS_ADP(36, num_classes, 4, auxiliary, genotype)

def DARTS_ADP_N4(num_classes, auxiliary=False):

  genotype = Genotype(
    normal=[
      ('max_pool_3x3', 0), 
      ('max_pool_3x3', 1), 
      ('max_pool_3x3', 0), 
      ('skip_connect', 2), 
      ('max_pool_3x3', 0), 
      ('max_pool_3x3', 2), 
      ('dil_conv_3x3', 4), 
      ('max_pool_3x3', 0)
    ], 
    normal_concat=range(2, 6), 
    reduce=[
      ('max_pool_3x3', 0), 
      ('max_pool_3x3', 1), 
      ('dil_conv_5x5', 2), 
      ('max_pool_3x3', 0), 
      ('sep_conv_5x5', 2), 
      ('max_pool_3x3', 0), 
      ('dil_conv_3x3', 2), 
      ('max_pool_3x3', 4)
    ], 
    reduce_concat=range(2, 6)
  )

  return DARTS_ADP(36, num_classes, 4, auxiliary, genotype)