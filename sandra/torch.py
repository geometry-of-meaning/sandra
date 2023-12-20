from typing import List, Union, Tuple

from sandra.situation_description import Description, DescriptionCollection, Situation, Element, Component

import torch
import torch.nn.functional as F

class ReasonerModule(torch.nn.Module):
  def __init__(self, ontology: DescriptionCollection, device=torch.device("cpu")):
    """
    Initialise the reasoner.

    Args:
        ontology (DescriptionCollection): Collection of descriptions 
          used by the reasoner to classify situations.
        device (optional): Device on which the reasoner module is loaded on. Defaults to cpu.
    """
    super().__init__()
    self.device = device
  
    self.ontology = ontology

    # compute the basis that spans the whole space by constructing
    # a matrix where the column are the encoding of each element
    self.basis = torch.stack([self.encode(e) for e in self.ontology.elements]).T.float()
    self.basis = F.normalize(self.basis, dim=0)

    self.description_bases = [
      self.basis[:, self.description_element_idxs(d)] for d in self.ontology.descriptions
    ]

    self.num_element_per_descriptions = torch.tensor([
      len(d.elements) for d in self.ontology.descriptions
    ])

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
    encoding = torch.tensor([self.g(e, e_i) for e_i in self.ontology.elements]).float()
    
    if type(e) == Description:
      # when encoding a description the basis of each element are also added
      encoding += torch.stack([self.encode_element(c) for c in e.elements]).sum(dim=0)

    return encoding.float()
    
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
    encoding = torch.zeros(len(self.ontology.elements), device=self.device)

    for c in s.components:
      if type(c) is Situation:
        encoding += self.encode_situation(c)
      else:
        c_idx = self.ontology.elements.index(c)
        encoding[c_idx] = 1

    return encoding

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
      raise ValueError(f"{e} is not of type Situation or Description")

  def forward(self, x: torch.tensor) -> torch.tensor:
    """
    Infer the descriptions that are satisfied by the encoded situation x.

    Args:
        x (torch.tensor): Situation array taken as input.
        
    Returns:
        torch.tensor: Array containing the distance between the input situation
          and each description. If distance is 0 then the description is
          completely satisfied.
    """
    if self.num_element_per_descriptions.device != self.device:
      self.num_element_per_descriptions = self.num_element_per_descriptions.to(self.device)
      
      for d_idx, d in enumerate(self.description_bases):
        self.description_bases[d_idx] = self.description_bases[d_idx].to(self.device)

    # turn x into batched if it is not
    x = torch.atleast_2d(x)
    
    # normalise x
    x = F.normalize(x, dim=-1)

    # in order to satisfy a description, a situation must be expressed
    # as a linear combination of the basis of such description
    # by solving the linear system Ab = x where A is the basis of a description,
    # and b is the situation, the solution x contains the coefficients
    # for each element in the description
    sat = torch.stack([torch.tensor([
     torch.heaviside(torch.linalg.lstsq(db, xi)[0], torch.tensor([0]).float()).sum() 
     for db_idx, db in enumerate(self.description_bases)]) / self.num_element_per_descriptions
     for xi in x])
    
    return sat
    