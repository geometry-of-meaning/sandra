from typing import List, Union, Tuple
from functools import partial
from math import log, pi, atan, exp

from sandra.ontology import KnowledgeBase
from functools import reduce

import torch
import torch.nn.functional as F

import torch
import torch.nn as nn
import math
from torch.autograd import Variable
from torch.autograd.function  import Function, InplaceFunction

import numpy as np

class StraighThroughHeaviside(Function):
  """
  Implements the straight through heaviside using the gradient estimation
  technique presented in [1].

  [1] https://arxiv.org/abs/1308.3432
  """
  def forward(self, input):
    """
    During the forward pass the regular heaviside function is used.
    The absolute value of the input values is considered. 
    A threshold is defined to make sure that small values are replaced by 0.
    """
    input = torch.heaviside(input, torch.zeros_like(input))
    return input
        
  def backward(self, grad_output):
    """
    During the backward pass, the gradient of the heaviside
    is approximated by simply copying the input gradients.
    """
    grad = F.hardtanh(grad_output, -1.0, 1.0)
    return grad, None, None, None

class StraighThroughGate(Function):
  """
  Implements the straight through heaviside using the gradient estimation
  technique presented in [1].

  [1] https://arxiv.org/abs/1308.3432
  """

  @staticmethod
  def forward(input, t):
    """
    During the forward pass the regular heaviside function is used.
    The absolute value of the input values is considered. 
    A threshold is defined to make sure that small values are replaced by 0.
    """
    mask = input > t

    out = torch.zeros_like(input)
    out[mask] = 1.0
    
    return out

  @staticmethod
  def setup_context(ctx, inputs, output):
    input, t = inputs
    ctx.t = t
        
  @staticmethod
  def backward(ctx, grad_output):
    """
    During the backward pass, the gradient of the heaviside
    is approximated by simply copying the input gradients.
    """
    grad = F.hardtanh(grad_output, -1.0, 1.0)
    return grad, None

class ReasonerModule(nn.Module):
  def __init__(self, kb: KnowledgeBase, device: str = "cpu", t: float = 0.01):
    super().__init__()
    self.kb = kb
    self.t = t
    
    # sorted attributes
    self._ordered_attrs = sorted(set(self.kb.attributes))
    
    # one-hot-encode auf
    self._onehot_attrs = torch.eye(len(self._ordered_attrs))

    # incorporate hierarchies
    self._encoded_attrs = torch.zeros_like(self._onehot_attrs)
    for i, x in enumerate(self._ordered_attrs):
      x_elements = self._onehot_attrs[[i] + [self._ordered_attrs.index(p) for p in self.kb.parents(x)]]
      self._encoded_attrs[i] = reduce(torch.logical_or, x_elements).to(self._encoded_attrs.dtype)
    
    # normalize the encodings
    self._encoded_attr = F.normalize(self._encoded_attrs, dim=1)

    # compute the frame encodings
    # to guarantee computational efficiency, a boolean matrix is used to keep track
    # of which component belong to which frame
    self._frame_gate = torch.zeros((len(self.kb.frames), len(self._ordered_attrs)))
    for i, f in enumerate(self.kb.frames):
      components_idxs = [self._ordered_attrs.index(c) for c in f.components]
      self._frame_gate[i, components_idxs] = 1.0
    self._frame_card = self._frame_gate.sum(dim=1)

  def forward(self, x):
    x = torch.atleast_2d(x)

    # normalize x
    x = F.normalize(x, dim=1)

    # retrieve attributes
    attrs = torch.einsum("bf,af->ba", x, self._encoded_attr)

    # piece-wise function
    # TODO: modularize this with tollerance
    inf = StraighThroughGate.apply(attrs, self.t)

    # aggregate frame-wise and normalize by number of elements in a frame
    inf = torch.einsum("ba,fa->bf", inf, self._frame_gate)
    inf = (inf + 1e-10) / (self._frame_card + 1e-10)

    return inf