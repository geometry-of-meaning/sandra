from typing import List, Union, Tuple
from sandra.ontology import Ontology, Situation, Element
import numpy as np

class Reasoner(object):
  def __init__(self, ontology: Ontology):
    """
    Initialise the reasoner.

    Args:
        ontology (Ontology): Ontology containing roles and descriptions
          used by the reasoner to classify situations.
    """
    super().__init__()
    self.ontology = ontology
    
    # cache encodings for faster execution
    self.__phi_cache = {}
    self.__encoding = {}
    
    # compute the basis that spans the whole space by constructing
    # a matrix where the column are the encoding of each element
    self.basis = np.stack([self.encode(e) for e in self.ontology.roles])
    self.basis = self.basis / np.linalg.norm(self.basis, axis=1).reshape(-1, 1)

    self.description_mask = np.zeros((len(self.ontology.descriptions), len(self.ontology.roles)))
    for d_idx, d in enumerate(self.ontology.descriptions):
      self.description_mask[d_idx, self.description_element_idxs(d)] = 1
      
    self.description_card = np.array([len(d.components) for d in self.ontology.descriptions])
    
  def description_element_idxs(self, d: Element) -> np.array:
    """
    Compute the indeces of the bases corresponding to element in the description.

    Args:
        d (Element): Input description.

    Returns:
        np.array: Elements' bases
    """
    idxs = set()
    for e in d.components:
      idxs.add(self.ontology.roles.index(e))
    
    return np.array(list(idxs))

  def phi(self, x: Element) -> np.array:
    """
    Computes phi for an element.

    Args:
        x (Element): Element to encode
    Returns:
        np.array: Vector of length (|C| + |D|) representing the encoded element.
    """
    if x.name not in self.__phi_cache:
      encoding = np.array([
        1 if x.name == y.name or y in x.descendants() else 0 
        for y in self.ontology.roles])
      self.__phi_cache[x.name] = encoding
    else:
      encoding = self.__phi_cache[x.name]
    
    return encoding

  def encode_element(self, d: Element) -> np.array:
    """
    Encode the provided element.
    
    Args:
        d (Element): The element that will be encoded.
    Returns:
        np.array: Vector of length (|C| + |D|) representing the encoded element.
    """
    if d.name in self.__encoding:
      encoding = self.__encoding[d.name]
    else:
      encoding = self.phi(d) + (0 if len(d.components) == 0 else np.vstack([self.encode_element(r) for r in d.components]).sum(axis=0))
      self.__encoding[d.name] = encoding

    return encoding

  def encode_situation(self, s: Situation) -> np.array:
    """
    Encode the provided situation.
    
    Args:
        s (Situation): The situation that will be encoded.
    Returns:
        np.array: Vector of length (|C| + |D|) representing the encoded situation.
    """
    return np.stack([self.encode(c) for c in s.individuals]).sum(axis=0)

  def encode(self, x: Union[Situation, Element]) -> np.array:
    """
    Encode the input by relying on description or element encoding.

    Args:
        x (Union[Situation, Element]): Element to be encoded.

    Raises:
        ValueError: Error raised if x is not a situation or a description.

    Returns:
        np.array: The input encoded as a vector.
    """
    if type(x) == Situation:
      return self.encode_situation(x)
    elif type(x) == Element:
      return self.encode_element(x)
    else:
      raise ValueError(f"{e} is not of type Situation or Description")

  def __call__(self, x: np.array) -> np.array:
    """
    Infer the descriptions that are satisfied by the encoded situation x.

    Args:
        x (np.array): Situation array taken as input.
        
    Returns:
        np.array: Array containing the distance between the input situation
          and each description. If distance is 0 then the description is
          completely satisfied.
    """
    # turn x into batched if it is not
    x = np.atleast_2d(x)

    # normalize x
    x = x / np.linalg.norm(x, axis=1).reshape(-1, 1)
    
    # in order to satisfy a description, a situation must be expressed
    # as a linear combination of the basis of such description
    # by solving the linear system Ab = x where A is the basis of a description,
    # and b is the situation, the solution x contains the coefficients
    # for each element in the description
    orthogonality = np.heaviside(x @ self.basis.T, np.zeros_like(x)) @ self.description_mask.T
    
    # compute the satisfied descriptions
    satisfied = orthogonality / self.description_card

    return satisfied
