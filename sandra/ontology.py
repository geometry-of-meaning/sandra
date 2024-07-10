from typing import List, Union, Dict, Tuple, Set
from itertools import chain
import rdflib
from urllib.parse import urlparse
from os import path

class Element(object):
  """
  An element is an abstraction of an element in the ontology.
  Any element in the ontology is always considered as predicate whose arity can be >= 1.
  It also defines a hierarchical structure: any element can be organized in a taxonomy.
  """
  def __init__(self, name: str):
    """
    Initialize an element

    Args:
        name (str): Name of the element.
    """
    self.name = name
    self.__parents = set()
    self.__children = set()
    self._components = set()
    self._component_of = set()

  @property
  def components(self) -> Set["Element"]:
    """
    Return the components (elements) of this element.

    Returns:
        Set[Element]: Set of elements of this element.
    """
    return self._components

  @property
  def parents(self) -> Set["Element"]:
    """
    Returns the parents of the current element in the taxonomy.

    Returns:
        Set[Element]: Set of parents of this element.
    """
    return self.__parents

  @property
  def children(self) -> Set["Element"]:
    """
    Returns the children of the current element in the taxonomy.

    Returns:
        Set[Element]: Set of children of this element.
    """
    return self.__children

  def add_parent(self, e: "Element"):
    """
    Add a parent to the element.

    Args:
        c (Element): Element added as parent.
    """
    if e != self:
      self.__parents.add(e)
      e.__children.add(self)
    
  def add_child(self, e: "Element"):
    """
    Add a child to the element.

    Args:
        e (Element): Element added as child.
    """
    if e != self:
      self.__children.add(e)
      e.__parents.add(self)

  def ancestors(self, visited = set()) -> List["Element"]:
    """
    Compute the parents of the current element up to the most general
    element available in the same hierarchy.

    Returns:
        List[Element]: List of ancestors of the current element.
    """
    if not hasattr(self, "__cached_ancestors"):
      visited.add(self)
      ancestors = set(self.parents)
      for p in self.parents:
        if p not in visited:
          ancestors = ancestors.union(p.ancestors(visited=visited))
      
      self.__cached_ancestors = ancestors
      
    return self.__cached_ancestors

  def descendants(self, visited = set()) -> List["Element"]:
    """
    Compute the children of the current element up to the most specific
    element available in the same hierarchy.

    Returns:
        List[Element]: List of descendants of the current element.
    """
    if not hasattr(self, "__cached_descendants"):
      visited.add(self)
      descendants = set(self.children)
      for c in self.children:
        if c not in visited:
          descendants = descendants.union(c.descendants(visited=visited))

      self.__cached_descendants = descendants
      
    return self.__cached_descendants

  def add(self, e: "Element"):
    """
    Adds a component of this element, transforming the element itself
    from a unary predicate to an n-ary predicate.

    Args:
        e (Element): Element added as component.
    """
    if e != self:
      self._components.add(e)
      e._component_of.add(self)

  @property
  def is_description(self) -> bool:
    """
    A description is an n-ary element and in the DnS framework is an entity that provides 
    the unity criterion to a "state of affairs".
    It partly represents a (possibly formalized) theory T (or one of its elements).
    It is a tuple formed by components defined by the theory T. 
    Examples of a description are a diagnosis, a climate change theory, etc.
    
    Differently than the original DnS formalization [1], a role can be an n-ary predicate.
    In that case, it is further composed of other roles and effectively acts as a description.
    Difference between a Role and a Description is kept to mantain coherency
    with the ones proposed by other frame semantics model, such as [2].

    [1] Gangemi, Aldo, and Peter Mika. "Understanding the semantic web through descriptions and situations." 
      OTM Confederated International Conferences" On the Move to Meaningful Internet Systems". 
      Berlin, Heidelberg: Springer Berlin Heidelberg, 2003.

    Returns:
        bool: Whether the element is a description.
    """
    return len(self.components) > 0

  @property
  def is_role(self) -> bool:
    """
    A role is an entity used to describe a state of affairs.
    Examples include a sample in a clinical data set, 
    a temperature, spatio-temporal coordinates etc.
    Roles are systematically related to descriptions in order to allow
    situations to satisfy some descriptions.
    
    [1] Gangemi, Aldo, and Peter Mika. "Understanding the semantic web through descriptions and situations." 
      OTM Confederated International Conferences" On the Move to Meaningful Internet Systems". 
      Berlin, Heidelberg: Springer Berlin Heidelberg, 2003.

    Returns:
        bool: Whether the element is a role.
    """
    return len(self.components) == 0 or len(self._component_of) > 0

  def __str__(self) -> str:
    """
    Returns:
        str: The role's name
    """
    return self.name

  def __repr__(self) -> str:
    """
    Returns:
        str: A string to represent the role
    """
    role_or_description = "Role" if self.is_role else "Description"
    return f"<{role_or_description}({str(self)})>"


class Situation(object):
  """
  A situation is intended as as a unitarian entity out of a "state of affairs", where 
  the unity criterion is provided by a Description.
  A state of affairs is a any non-empty set of assertions, representing a second-order entity.
  Examples include a clinical data set, a set of temperatures with spatio-temporal coordinates, etc.
  
  A situation must be systematically related to the roles of a description in order 
  to constitute a situation.
  It is composed of individuals, which have classified. The types of those roles is composed
  by the set of roles within the ontology.

  A situation satisfies a description when that description constitutes a valid interpretation
  that explains that situation.
  
  [1] Gangemi, Aldo, and Peter Mika. "Understanding the semantic web through descriptions and situations." 
    OTM Confederated International Conferences" On the Move to Meaningful Internet Systems". 
    Berlin, Heidelberg: Springer Berlin Heidelberg, 2003.
  """
  def __init__(self, individuals: List[Union[Element, "Situation"]] = []):
    """
    Create a situation. Optionally initialise it with the specified individuals.

    Args:
        individuals (List[Element | "Situation"] optional): 
          The initial individuals that are added to the situation. Defaults to [].
    """
    self.individuals = individuals

  def __str__(self) -> str:
    """
    Returns:
        str: The name of the situation
    """
    return f"{len(self.individuals)} individuals"

  def __repr__(self) -> str:
    """
    Returns:
        str: The string representation of the situation.
    """
    return f"<S({str(self)})>"

  def add(self, e: Union[Element, "Situation"]):
    """
    Add an individual to the situation.

    Args:
        e (Element | "Situation"): Element to add to the situation.
    """
    self.individuals.append(e)


class Ontology(dict):
  """
  A description collection is a class that acts as a collector of descriptions
  and provides methods to efficiently access a description or load collections
  descriptions from an OWL ontology serialised in RDF.
  """
  ROLES_QUERY = """
  PREFIX owl: <http://www.w3.org/2002/07/owl#>
  PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

  SELECT DISTINCT ?role WHERE {{ 
    ?role a <{role_class}> .
    MINUS {{ ?role rdfs:subClassOf/owl:someValuesFrom [] }} 
    FILTER (!ISBLANK(?role))
  }}
  """

  DESCRIPTIONS_QUERY = """
  PREFIX owl: <http://www.w3.org/2002/07/owl#>
  PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

  SELECT DISTINCT ?description ?role WHERE {{
    ?description rdfs:subClassOf/owl:someValuesFrom ?role .
    ?description a <{description_class}> .
    
    {{ ?role a <{role_class}> }}
    UNION
    {{ ?role a <{description_class}> }}
    
    FILTER (!ISBLANK(?description))
  }}
  """

  HIERARCHY_QUERY = """
    PREFIX owl: <http://www.w3.org/2002/07/owl#>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

    SELECT DISTINCT ?element ?parent ?parent WHERE {{
      ?element rdfs:subClassOf ?parent .

      {{ ?element a <{description_class}> }}
      UNION
      {{ ?element a <{role_class}> }}
      
      {{ ?parent a <{description_class}> }}
      UNION
      {{ ?parent a <{role_class}> }}
      
      FILTER (!ISBLANK(?element))
      FILTER (!ISBLANK(?parent))
    }}
    """

  @property
  def elements(self) -> List[Element]:
    """
    Returns:
        List[Element]: The list of elements contained in the ontology.
    """
    return [x for x in self.values()]

  @property
  def descriptions(self) -> List[Element]:
    """
    Returns:
        List[Description]: The list of descriptions contained in the collection.
    """
    return [x for x in self.values() if x.is_description]

  @property
  def roles(self) -> List[Element]:
    """
    Returns:
        List[Component]: The list of roles contained in the collection.
    """
    return [x for x in self.values() if x.is_role]
    
  def add(self, d: Element):
    """
    Add a description to the ontology.

    Args:
        d (Description): Description to add in the ontology.
    """
    if d.name not in self:
      self[d.name] = d

  @classmethod
  def from_graph(cls, 
    graph: Union[rdflib.Graph, str], 
    role_class: rdflib.URIRef = rdflib.OWL.Class,
    description_class: rdflib.URIRef = rdflib.OWL.Class) -> "Ontology":
    """
    Load an ontology of descriptions from an RDF file.

    Args:
        graph (rdflib.Graph): Input RDF graph.
        role_class (rdflib.URIRef, optional): Class used to define roles. Defaults to owl:Class.
        description_class (rdflib.URIRef, optional): Class used to define descriptions. Defaults to owl:Class.

    Returns:
        Ontology: Return the newly built collection of descriptions.
    """
    if type(graph) == str:
      graph = rdflib.Graph().parse(graph)

    ontology = cls() # create an empty description collection

    # extract all the roles from ontology
    for row in graph.query(cls.ROLES_QUERY.format(role_class=role_class)):
      ontology.add(Element(str(row.role)))

    # extract all the descriptions from the ontology
    descriptions_query_results = graph.query(cls.DESCRIPTIONS_QUERY.format(
      role_class=role_class,
      description_class=description_class))
    for row in descriptions_query_results:
      ontology.add(Element(str(row.description)))

    # add all the roles to each descriptions
    for row in descriptions_query_results:
      if str(row.description) in ontology and str(row.role) in ontology:
        ontology[str(row.description)].add(ontology[str(row.role)])

    # add the hierarchy
    for row in graph.query(cls.HIERARCHY_QUERY.format(
      role_class=role_class,
      description_class=description_class)):
      if str(row.element) in ontology and str(row.parent) in ontology:
        element = ontology[str(row.element)]
        parent = ontology[str(row.parent)]
        element.add_parent(parent)
        parent.add_child(element)

    return ontology

  def export(self, 
             base_iri: rdflib.Namespace = rdflib.Namespace("http://w3id.org/sandra/exported/"),
             role_class: str = "Role",
             description_class: str = "Description",
             property_name: str = "hasRole") -> rdflib.Graph:
    graph = rdflib.Graph()

    # munge iri for all the elements of the ontology
    iris = {
      el.name: base_iri[path.basename(urlparse(el.name).path)]
      for el in self.elements
    }

    for role in self.roles:
      # add to graph
      graph.add((iris[role.name], rdflib.RDF.type, rdflib.OWL.Class))
      graph.add((iris[role.name], rdflib.RDFS.subClassOf, base_iri[role_class]))

    # add descriptions
    for description in self.descriptions:
      graph.add((iris[description.name], rdflib.RDF.type, rdflib.OWL.Class))
      graph.add((iris[description.name], rdflib.RDFS.subClassOf, base_iri[description_class]))

      # add all the existential restrictions
      for component in description.components:
        # build existential restriction axiom
        restriction = rdflib.BNode()
        graph.add((restriction, rdflib.RDF.type, rdflib.OWL.Restriction))
        graph.add((restriction, rdflib.OWL.onProperty, base_iri[property_name]))
        graph.add((restriction, rdflib.OWL.someValuesFrom, iris[component.name]))

        # add it to the proper class
        graph.add((iris[description.name], rdflib.RDFS.subClassOf, restriction))

    return graph
