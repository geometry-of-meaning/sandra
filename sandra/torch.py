from typing import List, Union, Tuple
from functools import partial
from math import log, pi, atan, exp

from sandra.situation_description import Description, DescriptionCollection, Situation, Element, Component

import torch
import torch.nn.functional as F

class ReasonerModule(torch.nn.Module):
  def __init__(self, ontology: DescriptionCollection, epsilon: float = 0.1, device=torch.device("cpu")):
    """
    Initialise the reasoner.

    Args:
        ontology (DescriptionCollection): Collection of descriptions 
          used by the reasoner to classify situations.
        epsilon (float, optional): The epsilon value used to compute k and x_0 in the sigmoid function.
        device (optional): Device on which the reasoner module is loaded on. Defaults to cpu.
    """
    super().__init__()
    self.device = device
    self.ontology = ontology
    self.epsilon = epsilon

    # compute the basis that spans the whole space by constructing
    # a matrix where the column are the encoding of each element
    self.basis = torch.stack([self.encode(e) for e in self.ontology.elements])
    self.basis = F.normalize(self.basis).T

    self.A = torch.linalg.inv(self.basis)
    
    self.description_mask = torch.zeros((len(self.ontology.descriptions), len(self.ontology.elements)))
    for d_idx, d in enumerate(self.ontology.descriptions):
      self.description_mask[d_idx, self.description_element_idxs(d)] = 1
      # idxs = self.description_element_idxs(d)
      # A = self.basis[:, idxs]
      # self.description_bases.append(torch.linalg.pinv(A))
    self.description_card = self.description_mask.sum(dim=1)
        
    # self.A = torch.linalg.inv(self.basis).to(self.device)
    # # create a matrix that contains ones for element that are within a description
    # # and 0 for elements that are not within a description
    # # this will be used to index the correct elements for each description
    # self.description_mask = torch.zeros(
    #   (len(self.ontology.descriptions), len(self.ontology.elements)), 
    #   device=self.device)
    # for d_idx, d in enumerate(self.ontology.descriptions):
    #   if len(self.description_element_idxs(d)) > 0:
    #     self.description_mask[d_idx, self.description_element_idxs(d)] = 1
    
    # # keep track of the number of components in each description
    # self.components_per_descriptions = torch.tensor([
    #   len(d.elements) for d in self.ontology.descriptions
    # ])

    # self.descriptions_idxs = torch.tensor([
    #   self.ontology.elements.index(d) for d in self.ontology.descriptions
    # ])
    
    # compute the parameters for the sigmoid depending on epsilon
    #k = 10 #round((2 / epsilon) * log((1 - epsilon) / epsilon), 1)
    #x0 = 0.5 #round((3 / 2) * epsilon, 1)
    #print(k, x0)
    #self.sigma = partial(lambda x: 1 / (1 + torch.exp(-1 * (x - x0))))
    #self.sigma = partial(lambda x: 0.5 + (1/pi) * torch.atan((x - 0.01) / 0.01))
    #https://arxiv.org/pdf/2102.07156.pdf
    #self.gamma = 4

  def description_element_idxs(self, d: Description) -> torch.tensor:
    """
    Compute the indeces of the bases corresponding to element in the description.

    Args:
        d (Description): Input description.

    Returns:
        torch.tensor: Elements' bases
    """
    idxs = set()
    for e in d.elements:
      idxs.add(self.ontology.elements.index(e))

      for a in e.ancestors():
        if type(a) is Description:
          idxs = idxs.union(self.description_element_idxs(a))
    
    return torch.tensor(list(idxs))

  def g(self, x: Element, y: Element) -> int:
    """
    Assigns to each pair of elements a value based their relationship.
    
    g(x, y) = {
      1 if x == y
      0 otherwise
    }

    Args:
        x (Element): Element x
        y (Element): Element y
    """
    return 1 if x == y or y in x.descendants() else 0

  def encode_element(self, e: Element) -> torch.tensor:
    """
    Encode the provided element using the f function.

    f(e) = [ g(e, c_0), ..., g(e, c_n), g(e, d_0), ..., g(e, d_n) ] 
      for c_0 ... c_n in C and d_0 ... d_n in D

    Args:
        e (Element): The element that will be encoded.
    Returns:
        torch.tensor: Tensor of length (|C| + |D|) representing the encoded component.
    """
    return torch.tensor([self.g(e, e_i) for e_i in self.ontology.elements]).float()
    
  def encode_situation(self, s: Situation) -> torch.tensor:
    """
    Encode the provided situation using the f~ function.

    f~(s) = [ g~(e, c_0), ..., g~(e, c_n), g~(e, d_0), ..., g~(e, d_n) ] 
      for c_0 ... c_n in C and d_0 ... d_n in D

    Args:
        s (Situation): The situation that will be encoded.
    Returns:
        torch.tensor: Tensor of length (|C| + |D|) representing the encoded situation.
    """
    return torch.stack([self.encode(c) for c in s.components]).sum(dim=0)

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
    elif type(x) == Description or type(x) == Component:
      return self.encode_element(x)
    else:
      raise ValueError(f"{x} is not of type Situation or Description")
  
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
      #self.components_per_descriptions = self.components_per_descriptions.to(self.device)
      #self.description_mask = self.description_mask.to(self.device)
      #self.descriptions_idxs = self.descriptions_idxs.to(self.device)
      #self.basis = self.basis.to(self.device)
      self.A = self.A.to(self.device)
      self.description_card = self.description_card.to(self.device)
      self.description_mask = self.description_mask.to(self.device)
      #self.description_bases = [d.to(self.device) for d in self.description_bases]
      #self.description_bases = self.description_bases.to(self.device)

    # turn x into batched if it is not
    x = torch.atleast_2d(x)
    x = F.normalize(x)

    # in order to satisfy a description, a situation must be expressed
    # as a linear combination of the basis of such description
    # by solving the linear system Ax = y where A is the basis of a description,
    # and x is the situation, the solution y contains the coefficients
    # for each element in the description
    if differentiable:
      # h = lambda z: 1 - torch.exp(-32 * z) + z * exp(-32)
      h = lambda x: F.hardtanh(x, min_val=0.0, max_val=1.0)
    else:
      h = lambda x: torch.heaviside(x, torch.zeros_like(x))
  
    coefficients = torch.abs(self.A.T @ x.T).T
    satisfied = h(coefficients) @ self.description_mask.T
    satisfied = satisfied / self.description_card

    # satisfied = torch.stack([
    #  h(torch.abs(d @ x.T)).sum(dim=0) / len(d)
    #  for d in self.description_bases
    # ]).T
    
    return satisfied
    #return torch.abs(x @ self.basis)