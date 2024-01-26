from typing import List, Union, Tuple
from sandra.situation_description import Description, DescriptionCollection, Situation, Component, Element
import numpy as np

class Reasoner(object):
  def __init__(self, ontology: DescriptionCollection):
    """
    Initialise the reasoner.

    Args:
        ontology (DescriptionCollection): Collection of components,
          and descriptions used by the reasoner to classify situations.
    """
    super().__init__()
    self.ontology = ontology

    # compute the basis that spans the whole space by constructing
    # a matrix where the column are the encoding of each element
    self.basis = np.stack([self.encode(e) for e in self.ontology.elements]).T
    self.basis = self.basis / np.linalg.norm(self.basis, axis=0)

    self.A = np.linalg.inv(self.basis)

    self.description_mask = np.zeros((len(self.ontology.descriptions), len(self.ontology.elements)))
    for d_idx, d in enumerate(self.ontology.descriptions):
      self.description_mask[d_idx, self.description_element_idxs(d)] = 1
      
    self.description_card = self.description_mask.sum(axis=1)

  def description_element_idxs(self, d: Description) -> np.array:
    """
    Compute the indeces of the bases corresponding to element in the description.

    Args:
        d (Description): Input description.

    Returns:
        np.array: Elements' bases
    """
    idxs = set()
    for e in d.elements:
      idxs.add(self.ontology.elements.index(e))

      for a in e.ancestors():
        if type(a) is Description:
          idxs = idxs.union(self.description_element_idxs(a))
    
    return np.array(list(idxs))

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

  def encode_element(self, e: Element) -> np.array:
    """
    Encode the provided element using the f function.

    f(e) = [ g(e, c_0), ..., g(e, c_n), g(e, d_0), ..., g(e, d_n) ] 
      for c_0 ... c_n in C and d_0 ... d_n in D

    Args:
        e (Element): The element that will be encoded.
    Returns:
        np.array: Vector of length (|C| + |D|) representing the encoded component.
    """
    return np.array([self.g(e, e_i) for e_i in self.ontology.elements])
    
  def encode_situation(self, s: Situation) -> np.array:
    """
    Encode the provided situation using the f~ function.

    f~(s) = [ g~(e, c_0), ..., g~(e, c_n), g~(e, d_0), ..., g~(e, d_n) ] 
      for c_0 ... c_n in C and d_0 ... d_n in D

    Args:
        s (Situation): The situation that will be encoded.
    Returns:
        np.array: Vector of length (|C| + |D|) representing the encoded situation.
    """
    return np.stack([self.encode(c) for c in s.components]).sum(axis=0)

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
    elif type(x) == Description or type(x) == Component:
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
    
    # normalise x
    x = x / np.linalg.norm(x, axis=-1).reshape(-1, 1)

    # in order to satisfy a description, a situation must be expressed
    # as a linear combination of the basis of such description
    # by solving the linear system Ab = x where A is the basis of a description,
    # and b is the situation, the solution x contains the coefficients
    # for each element in the description
    coefficients = np.abs(self.A.T @ x.T).T
    satisfied = np.heaviside(coefficients, np.zeros_like(coefficients)) @ self.description_mask.T
    satisfied = satisfied / self.description_card

    return satisfied  