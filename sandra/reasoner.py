from typing import List, Union, Tuple

from itertools import chain
from sandra.situation_description import Description, DescriptionCollection, Situation, Component, Element

import numpy as np
import warnings

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

    self.description_bases = [
      self.basis[:, self.description_element_idxs(d)] for d in self.ontology.descriptions
    ]

    self.num_element_per_descriptions = np.array([
      len(d.elements) for d in self.ontology.descriptions
    ])

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
    #return 1 if x == y else 0

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
    encoding = np.array([self.g(e, e_i) for e_i in self.ontology.elements])
    
    if type(e) == Description:
      # when encoding a description the basis of each element are also added
      encoding += np.sum([self.encode_element(c) for c in e.elements], axis=0)

    return encoding
    
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
    encoding = np.zeros(len(self.ontology.elements))

    for c in s.components:
      if type(c) is Situation:
        encoding += self.encode_situation(c)
      else:
        c_idx = self.ontology.elements.index(c)
        encoding[c_idx] = 1

    return encoding
    #return np.sum([self.encode(e) for e in s.components], axis=0)

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

  def infer(self, x: np.array) -> np.array:
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
    sat = np.stack([np.array([
      np.heaviside(np.linalg.lstsq(db, xi, rcond=None)[0], 0).sum() 
      for db_idx, db in enumerate(self.description_bases)]) / self.num_element_per_descriptions
      for xi in x])

    return sat
    
  def classify(self, s: Situation) -> List[Tuple[Description, float]]:
    """
    Classify an input situation and return the probability that it
    satisfied each class.

    Args:
        s (Situation): Input situation.

    Returns:
        List[Tuple[Description, float]]: List of tuples of the type
          (description, score) where description is one of the 
          descriptions on the ontology and score is the probability
          that the situation satisfies a description.
    """
    encoded_s = self.encode(s)
    p = self.infer(encoded_s)[0]
    return list(zip(self.ontology.descriptions, p))
    