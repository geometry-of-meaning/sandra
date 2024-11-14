from typing import List, Union, Tuple
from sandra.ontology import KnowledgeBase
import numpy as np

from functools import reduce

class Reasoner(object):
  def __init__(self, kb: KnowledgeBase):
    """
    Initialise the reasoner.

    Args:
        ontology (Ontology): Ontology containing roles and descriptions
          used by the reasoner to classify situations.
    """
    super().__init__()
    self.kb = kb
    
    # sorted attributes and frames set
    # TODO: Attribute and Frame should not be in AuF!
    self._auf = sorted(set(self.kb.attributes).union(self.kb.frames))
    
    # one-hot-encode auf
    self._onehot_auf = np.eye(len(self._auf))

    # incorporate hierarchies
    self._encoded_auf = np.zeros_like(self._onehot_auf)
    for i, x in enumerate(self._auf):
      x_elements = self._onehot_auf[[i] + [self._auf.index(p) for p in self.kb.parents(x)]]
      self._encoded_auf[i] = reduce(np.logical_or, x_elements).astype(self._encoded_auf.dtype)
    
    # normalize the encodings
    self._encoded_auf /= np.linalg.norm(self._encoded_auf, axis=1)

    # compute the frame encodings
    # to guarantee computational efficiency, a boolean matrix is used to keep track
    # of which component belong to which frame
    self._frame_gate = np.zeros((len(self.kb.frames), len(self._auf)))
    for i, f in enumerate(self.kb.frames):
      components_idxs = [self._auf.index(c) for c in f.components]
      self._frame_gate[i, components_idxs] = 1.0
    self._frame_card = self._frame_gate.sum(axis=1)

  def infer(self, x):
    x = np.atleast_2d(x)

    # normalize x
    x /= np.linalg.norm(x, axis=1)

    # retrieve attributes
    attrs = np.einsum("bf,af->ba", x, self._encoded_auf) ** 2

    # piece-wise function
    # TODO: modularize this with tollerance
    inf = np.heaviside(attrs, np.zeros_like(attrs))

    # aggregate frame-wise and normalize by number of elements in a frame
    inf = np.einsum("ba,fa->bf", inf, self._frame_gate)
    inf = inf / self._frame_card

    return inf