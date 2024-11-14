from typing import List, Union, Dict, Tuple, Set
from itertools import chain
from collections import defaultdict
import rdflib
from urllib.parse import urlparse
from os import path
from urllib.parse import quote_plus

from dataclasses import dataclass

BASE_IRI = rdflib.Namespace("https://w3id.org/geometryofmeaning/sandra#")
DEFAULT_FRAME_IRI = BASE_IRI["Frame"]
DEFAULT_ATTRIBUTE_IRI = BASE_IRI["Attribute"]
DEFAULT_COMPONENT_IRI = BASE_IRI["hasComponent"]

@dataclass
class Element(object):
  """
  An abstraction of an element in the Knowledge Base.
  """
  name: str

  def __repr__(self) -> str:
    """
    Returns:
        str: A string to represent the role
    """
    return f"Element<{self.name}>"

  def __hash__(self) -> int:
    return hash(self.name)

  def __lt__(self, b: "Element"):
    return self.name < b.name

  def __gt__(self, b: "Element"):
    return self.name < b.name


@dataclass
class Attribute(Element):
  """
  An attribute on the Knowledge Base.
  """
  def __hash__(self) -> int:
    return hash(self.name)


@dataclass
class Frame(Element):
  """
  A frame in the Knowledge Base.
  """
  components: list[Attribute]

  def __hash__(self) -> int:
    return hash(self.name)



def parse_attributes_taxonomy(graph, node, nodes = {}, parents = defaultdict(list)):
  if node not in nodes:
      # strip iri
      nodes[node] = Attribute(str(node))

  for child, _, _ in graph.triples((None, rdflib.RDFS.subClassOf, node)):
    nodes, parents = parse_attributes_taxonomy(graph, child, nodes, parents)
    parents[nodes[child]].append(nodes[node])
  
  return nodes, parents

def parse_frames_taxonomy(graph, attributes, node, r_iri, nodes = {}, parents = defaultdict(list)):
  if node not in nodes:
    # extract the components of this node
    components = graph.objects(node, rdflib.RDFS.subClassOf)
    components = filter(lambda c: next(graph.objects(c, rdflib.RDF.type)) == rdflib.OWL.Restriction, components)
    components = filter(lambda c: next(graph.objects(c, rdflib.OWL.onProperty)) == r_iri, components)
    components = map(lambda c: next(graph.objects(c, rdflib.OWL.someValuesFrom)), components)
    components = [attributes[a] for a in components]
    
    # TODO: strip iri
    nodes[node] = Frame(str(node), components)

  for child, _, _ in graph.triples((None, rdflib.RDFS.subClassOf, node)):
    nodes, parents = parse_frames_taxonomy(graph, attributes, child, r_iri, nodes, parents)
    parents[nodes[child]].append(nodes[node])
  
  return nodes, parents

@dataclass
class KnowledgeBase(object):
  """
  A Knowledge Base is defined as a finite set of attributes and frames.
  It also includes hierarchical relations between attributes and frames.
  """
  attributes: list[Attribute]
  attributes_parents: dict[Attribute, list[Attribute]]

  frames: list[Frame]
  frames_parents: dict[Frame, list[Frame]]

  def parents(self, x):
    match x:
      case Attribute():
        parents = set(self.attributes_parents[x])
      case Frame():
        parents = set(self.frames_parents[x])
    
    collected_parent = parents.copy()
    for p in parents:
      collected_parent.add(p)
      collected_parent.update(self.parents(p))

    return collected_parent

  def __len__(self) -> int:
    return len(self.attributes) + len(self.frames)

  def __repr__(self) -> str:
    return f"KnowledgeBase({len(self.attributes)} attributes, {len(self.frames)} frames)"

  def to_graph(self,
    base_iri: rdflib.Namespace | str = BASE_IRI,
    attribute_iri: rdflib.URIRef | str = DEFAULT_ATTRIBUTE_IRI,
    frame_iri: rdflib.URIRef | str = DEFAULT_FRAME_IRI,
    r_iri: rdflib.URIRef | str = DEFAULT_COMPONENT_IRI) -> rdflib.Graph:
    
    graph = rdflib.Graph()
    graph.add((r_iri, rdflib.RDF.type, rdflib.OWL.ObjectProperty))
    
    for a in self.attributes:
      name = quote_plus(a.name)

      graph.add((base_iri[name], rdflib.RDF.type, rdflib.OWL.Class))
      graph.add((base_iri[name], rdflib.RDFS.subClassOf, attribute_iri))

    return graph
    

  @staticmethod
  def from_graph(
    graph: rdflib.Graph, 
    attribute_iri: rdflib.URIRef | str = DEFAULT_ATTRIBUTE_IRI,
    frame_iri: rdflib.URIRef | str = DEFAULT_FRAME_IRI,
    r_iri: rdflib.URIRef | str = DEFAULT_COMPONENT_IRI):
    hierarchy_query = "SELECT * WHERE {{ ?x rdfs:subClassOf* <{iri}> . FILTER (?x != <{iri}>). }}"
    components_query = """
    SELECT * WHERE {{ 
      <{frame_iri}> rdfs:subClassOf [ 
        owl:onProperty <{r_iri}> ; 
        owl:someValuesFrom ?x ] .
      FILTER (?x != <{iri}>). 
    }}"""

    # extract attributes
    attributes = {
      x: Attribute(name=str(x)) # TODO: strip iri
      for x, *_ in graph.query(hierarchy_query.format(iri=attribute_iri))
    }

    # extract attribute hierarchy
    attributes_parents = {
      attribute: [attributes[i] for i in graph.objects(iri, rdflib.RDFS.subClassOf) if i in attributes]
      for iri, attribute in attributes.items()
    }

    # extract frames
    frames = {}
    for iri, *_ in graph.query(hierarchy_query.format(iri=frame_iri)):
      components = [
        attributes[c]
        for c, *_ in graph.query(components_query.format(frame_iri=iri, r_iri=r_iri, iri=frame_iri))
      ] 

      # TODO: strip iri
      frames[iri] = Frame(str(iri), components=components)

    # extract frames hierarchy
    frames_parents = {
      frame: [frames[i] for i in graph.objects(iri, rdflib.RDFS.subClassOf) if i in frames]
      for iri, frame in frames.items()
    }


    return KnowledgeBase(
      attributes=list(attributes.values()),
      attributes_parents=attributes_parents,
      frames=list(frames.values()),
      frames_parents=frames_parents
    )