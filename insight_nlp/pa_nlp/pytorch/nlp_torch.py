#author: Tian Xia (SummerRainET2008@gmail.com)

from pa_nlp import *
from pa_nlp.nlp import Logger
from pa_nlp import nlp
from pa_nlp.pytorch import *
import torch

class Dense(nn.Module):
  def __init__(self, out_dim, activation=None):
    super(Dense, self).__init__()
    self._out_dim = out_dim
    self._activation = activation if activation is not None else lambda x: x
    self._net = None

  def __call__(self, x: torch.Tensor):
    if self._net is None:
      in_dim = x.shape[-1]
      self._net = nn.Linear(in_dim, self._out_dim)

    out = self._net(x)
    out = self._activation(out)

    return out

