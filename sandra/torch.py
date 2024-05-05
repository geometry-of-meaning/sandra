from typing import List, Union, Tuple
from functools import partial
from math import log, pi, atan, exp

from sandra.ontology import Ontology, Situation, Element

import torch
import torch.nn.functional as F

class ReasonerModule(torch.nn.Module):
  def __init__(self, ontology: Ontology, epsilon: float = 128, device=torch.device("cpu")):
    """
    Initialise the reasoner.

    Args:
        ontology (Ontology): Ontology containing roles and descriptions
          used by the reasoner to classify situations.
        epsilon (float, optional): Controls the degree of approximation of the smooth Heaviside 
          according to the function defined in [1]. Defaults to 128.
        device (optional): Device on which the reasoner module is loaded on. Defaults to cpu.
    
    [1] Tiwari, R. et al., "ChipNet: Budget-Aware Pruning with Heaviside Continuous Approximations", ICLR 2021
    """
    super().__init__()
    self.device = device
    self.ontology = ontology
    self.epsilon = epsilon

    # cache encodings for faster execution
    self.__phi_cache = {}
    self.__encoding = {}
    
    # compute the basis that spans the whole space by constructing
    # a matrix where the column are the encoding of each element
    self.basis = torch.stack([self.encode(e) for e in self.ontology.elements])
    self.basis = F.normalize(self.basis)

    self.A = torch.linalg.pinv(self.basis)
    
    self.description_mask = torch.zeros((len(self.ontology.descriptions), len(self.ontology.elements)))
    for d_idx, d in enumerate(self.ontology.descriptions):
      self.description_mask[d_idx, self.description_element_idxs(d)] = 1
      
    self.description_card = torch.tensor([len(d.components) for d in self.ontology.descriptions])
        
  def description_element_idxs(self, d: Element) -> torch.tensor:
    """
    Compute the indeces of the bases corresponding to element in the description.

    Args:
        d (Element): Input description.

    Returns:
        torch.tensor: Elements' bases
    """
    idxs = set()
    for e in d.components:
      idxs.add(self.ontology.elements.index(e))
    
    return torch.tensor(list(idxs))

  def phi(self, x: Element) -> torch.Tensor:
    """
    Computes phi for an element.

    Args:
        x (Element): Element to encode
    Returns:
        torch.Tensor: Vector of length (|C| + |D|) representing the encoded.
    """
    if x.name not in self.__phi_cache:
      encoding = torch.Tensor([
        1 if x == y or y in x.descendants() else 0 
        for y in self.ontology.elements])
      self.__phi_cache[x.name] = encoding
    else:
      encoding = self.__phi_cache[x.name]
    
    return encoding

  def encode_element(self, d: Element) -> torch.Tensor:
    """
    Encode the provided element.
    
    Args:
        e (Element): The element that will be encoded.
    Returns:
        np.Tensor: Vector of length (|C| + |D|) representing the encoded element.
    """
    if d.name in self.__encoding:
      encoding = self.__encoding[d.name]
    else:
      encoding = self.phi(d) + (0 if len(d.components) == 0 else torch.vstack([self.phi(r) for r in d.components]).sum(axis=0))
      self.__encoding[d.name] = encoding
    return encoding
  
  def encode_situation(self, s: Situation) -> torch.tensor:
    """
    Encode the provided situation.

    Args:
        s (Situation): The situation that will be encoded.
    Returns:
        torch.tensor: Tensor of length (|C| + |D|) representing the encoded situation.
    """
    return torch.stack([self.encode(c) for c in s.individuals]).sum(dim=0)

  def encode(self, x: Union[Situation, Element]) -> torch.tensor:
    """
    Encode the input by relying on description or element encoding.

    Args:
        x (Union[Situation, Element]): Element to be encoded.

    Raises:
        ValueError: Error raised if x is not a situation or a description.

    Returns:
        torch.tensor: The input encoded as a tensor.
    """
    if type(x) == Situation:
      return self.encode_situation(x)
    elif type(x) == Element:
      return self.encode_element(x)
    else:
      raise ValueError(f"{e} is not of type Situation or Description")
  
  def forward(self, x: torch.tensor, differentiable=True) -> torch.tensor:
    """
    Infer the descriptions that are satisfied by the encoded situation x.

    Args:
        x (torch.tensor): Situation array taken as input.
        
    Returns:
        torch.tensor: Array containing the distance between the input situation
          and each description. If distance is 0 then the description is
          completely satisfied.
    """
    if self.A.device != self.device:
      self.A = self.A.to(self.device)
      self.description_card = self.description_card.to(self.device)
      self.description_mask = self.description_mask.to(self.device)
      
    # turn x into batched if it is not
    x = torch.atleast_2d(x)
    x = F.normalize(x)

    # in order to satisfy a description, a situation must be expressed
    # as a linear combination of the basis of such description
    # by solving the linear system Ax = y where A is the basis of a description,
    # and x is the situation, the solution y contains the coefficients
    # for each element in the description
    if differentiable:
      h = lambda z: 1 - torch.exp(-self.epsilon * z) + z * exp(-self.epsilon)
    else:
      h = lambda x: torch.heaviside(x, torch.zeros_like(x))
  
    coefficients = self.A @ x.T
    # remove coefficients smaller than 1e-5
    coefficients = torch.abs(coefficients).T.clamp(1e-5, 1e5) - 1e-5
    
    satisfied = h(coefficients) @ self.description_mask.T
    satisfied = satisfied / self.description_card

    return satisfied

  def classify_in_subspace(self, x: torch.Tensor, subspace: torch.Tensor, differentiable: bool = True):
    if self.A.device != self.device:
      self.A = self.A.to(self.device)
      self.description_card = self.description_card.to(self.device)
      self.description_mask = self.description_mask.to(self.device)
      
    card = subspace.sum(axis=1) + 1e-6
    
    x = torch.atleast_2d(x)
    x = F.normalize(x)
    if differentiable:
      h = lambda z: 1 - torch.exp(-self.epsilon * z) + z * exp(-self.epsilon)
      #h = F.relu
    else:
      h = lambda x: torch.heaviside(x, torch.zeros_like(x))
  
    coefficients = self.A @ x.T
    coefficients = torch.abs(coefficients).T.clamp(1e-5, 1e5) - 1e-5
    satisfied = h(coefficients) @ subspace.T
    satisfied = satisfied / card
    
    return satisfied