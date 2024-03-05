from typing import List, Union, Dict, Tuple
from itertools import chain
import rdflib

class Element(object):
  """
  An element is an abstraction of an element in the ontology,
  which includes a hierarchical structure.
  """
  def __init__(self):
    self.parents = set()
    self.children = set()

  def add_parent(self, e: "Element"):
    """
    Add a parent to the element.

    Args:
        c (Element): Element added as parent.
    """
    if e != self:
      self.parents.add(e)
      e.children.add(self)
    
  def add_child(self, e: "Element"):
    """
    Add a child to the element.

    Args:
        e (Element): Element added as child.
    """
    if e != self:
      self.children.add(e)
      e.parents.add(self)

  def ancestors(self) -> List["Element"]:
    """
    Compute the parents of the current element up to the most general
    element available in the same hierarchy.

    Returns:
        List[Element]: List of ancestors of the current element.
    """
    if not hasattr(self, "__cached_ancestors"):
      ancestors = set()
      for p in self.parents:
        if p != self and p not in ancestors:
          ancestors = ancestors.union(p.ancestors())

      ancestors = ancestors.union(self.parents)
      self.__cached_ancestors = ancestors
      
    return self.__cached_ancestors

  def descendants(self) -> List["Element"]:
    """
    Compute the children of the current element up to the most specific
    element available in the same hierarchy.

    Returns:
        List[Element]: List of descendants of the current element.
    """
    if not hasattr(self, "__cached_descendants"):
      descendants = set()
      for c in self.children:
        if c != self and c not in descendants:
          descendants = descendants.union(c.descendants())

      descendants = descendants.union(self.children)
      self.__cached_descendants = descendants
      
    return self.__cached_descendants

  # @property
  # def closure(self) -> List["Element"]:
  #   """
  #   Returns:
  #       List[Element]: Computes the closure of the current element by collecting
  #       all the ancestors and the descendants of this element.
  #   """
  #   closure = set()

  #   if len(self.parents) > 0:
  #     closure = closure.union(self.parents)
  #     closure = closure.union(*[p.ancestors() for p in self.parents])

  #   if len(self.children) > 0:
  #     closure = closure.union(self.children)
  #     closure = closure.union(*[c.descendants() for c in self.children])

  #   return closure


class Component(Element):
  """
  A component is an entity used to describe a state of affairs.
  Examples include a sample in a clinical data set, 
  a temperature, spatio-temporal coordinates etc.
  Components are systematically related to descriptions in order to allow
  situations to be described by some descriptions.

  In [1] a component is formalised as the predicate `P_i`. 

  [1] Gangemi, Aldo, and Peter Mika. "Understanding the semantic web through descriptions and situations." 
    OTM Confederated International Conferences" On the Move to Meaningful Internet Systems". 
    Berlin, Heidelberg: Springer Berlin Heidelberg, 2003.
  """
  def __init__(self, name: str):
    """
    Create a component.

    Args:
        name (str): Name of the component.
    """
    super().__init__()
    self.name = name
    self.parents = set()
    self.children = set()
  
  def __str__(self) -> str:
    """
    Returns:
        str: The component name
    """
    return self.name
  
  def __repr__(self) -> str:
    """
    Returns:
        str: A string to represent the component
    """
    return f"<Component({str(self)})>"


class Description(Element):
  """
  A description in the D&S framework is an entity that provides the unity criterion to
  a "state of affairs".
  It partly represents a (possibly formalized) theory T (or one of its elements).
  It is a tuple formed by components defined by the theory T. 
  Examples of a description are a diagnosis, a climate change theory, etc.
  
  Each Description can be considered a component and used to form other descriptions. 
  In [1] this is represented by the predicate `P_i^D` and the axiom
  `selects(x, y) -> P_i^D(x), P_i(y)` is used to bind the description represented
  using such component.

  [1] Gangemi, Aldo, and Peter Mika. "Understanding the semantic web through descriptions and situations." 
    OTM Confederated International Conferences" On the Move to Meaningful Internet Systems". 
    Berlin, Heidelberg: Springer Berlin Heidelberg, 2003.
  """
  def __init__(self, name: str):
    """
    Initialise an empty description.

    Args:
        name (str): Name of the description.
    """
    super().__init__()
    self.name = name
    self.elements = set()

  def __str__(self) -> str:
    """
    Returns:
        str: The name of the description
    """
    return self.name

  def __repr__(self) -> str:
    """
    Returns:
        str: The string representation of the description.
    """
    return f"<D({str(self)})>"

  def add_element(self, c: Element):
    """
    Add a component to the class.

    Args:
        c (Element): The element added as a component.
    """
    self.elements.add(c)

  @property
  def closure(self) -> List[Element]:
    """
    Returns:
        List[Element]: Computes the closure of the current element by collecting
        all the ancestors and the descendants of this description and of the
        elements it is composed of.
    """
    closure = super().closure
    
    # add the closure of each element
    if len(self.elements) > 0:
      closure = closure.union(self.elements)

      for e in self.elements:
        closure = closure.union(e.closure)

    return closure


class Situation(object):
  """
  A situation is intended as as a unitarian entity out of a "state of affairs", where 
  the unity criterion is provided by a Description.
  A state of affairs is a any non-empty set of assertions, representing a second-order entity.
  Examples include a clinical data set, a set of temperatures with spatio-temporal coordinates, etc.
  
  A situation must be systematically related to the components of a description in order 
  to constitute a situation.

  In [1] a situation is defined as `S(x)`. The components of the situations are asserted
  through the `settingFor(x, y)` where `x` is the situation and `y` is the component.
  The element of a situation can also be another situation. This allows the composition
  of situation. 
  In [1] this is expressed through the axiom `forall S(x) -> forall part(x, y) -> S(y)`.

  Finally, a situation satisfies a description when the interpretation of the situation
  as a description (defined as s-descriptio or situation-description in [1]) 
  overlaps with the one of a description.
  In [1] this is expressed through the axiom `satisfies(x, y) -> S(x), SD(x)` and the
  axiom `SD(x) -> D(x)`.
  The axiom 
  `settingFor(x, y) -> 
    exists zw P_i^D(z), 
              SD(w), 
              t_component(w, z), 
              satisfies(x, w),
              select(z, y)
  `
  states that if a situation (x) has a s-description (y) component then the situation 
  must also satisfy the s-description (y).
  In other words, if a description D includes another description D' among its components,
  the situation needs to satisfy D' as well.
  This allows nested situations to satisfy nested descriptions.

  [1] Gangemi, Aldo, and Peter Mika. "Understanding the semantic web through descriptions and situations." 
    OTM Confederated International Conferences" On the Move to Meaningful Internet Systems". 
    Berlin, Heidelberg: Springer Berlin Heidelberg, 2003.
  """
  def __init__(self, components: List[Union[Element, "Situation"]] = []):
    """
    Create a situation. Optionally initialise it with the specified components.

    Args:
        components (List[Element | "Situation"] optional): 
          The initial components that are added to the situation. Defaults to [].
    """
    self.components = components

  def __str__(self) -> str:
    """
    Returns:
        str: The name of the situation
    """
    return "Situation"

  def __repr__(self) -> str:
    """
    Returns:
        str: The string representation of the situation.
    """
    return f"<S({str(self)})>"

  def add(self, e: Union[Element, "Situation"]):
    """
    Add an element to the situation.

    Args:
        e (Element | "Situation"): Element to add to the situation.
    """
    self.components.append(e)


class DescriptionCollection(dict):
  """
  A description collection is a class that acts as a collector of descriptions
  and provides methods to efficiently access a description or load collections
  descriptions from an OWL ontology serialised in RDF.
  """
  COMPONENTS_QUERY = """
  PREFIX owl: <http://www.w3.org/2002/07/owl#>
  PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

  SELECT DISTINCT ?component WHERE {{ 
    ?component a <{component_class}> .
    MINUS {{ ?component rdfs:subClassOf/a owl:Restriction }} 
    FILTER (!ISBLANK(?component))
  }}
  """

  DESCRIPTIONS_QUERY = """
  PREFIX owl: <http://www.w3.org/2002/07/owl#>
  PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

  SELECT DISTINCT ?description ?element WHERE {{
    ?description rdfs:subClassOf/owl:someValuesFrom ?element .
    ?description a <{description_class}> .
    
    {{ ?element a <{component_class}> }}
    UNION
    {{ ?element a <{description_class}> }}
    
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
      {{ ?element a <{component_class}> }}
      
      {{ ?parent a <{description_class}> }}
      UNION
      {{ ?parent a <{component_class}> }}
      
      FILTER (!ISBLANK(?element))
      FILTER (!ISBLANK(?parent))
    }}
    """

  @property
  def elements(self) -> List[Element]:
    """
    Returns:
        List[Component]: The list of elements contained in the collection.
    """
    return [x for x in self.values()]

  @property
  def descriptions(self) -> List[Description]:
    """
    Returns:
        List[Description]: The list of descriptions contained in the collection.
    """
    return [x for x in self.values() if type(x) is Description]

  @property
  def components(self) -> List[Component]:
    """
    Returns:
        List[Component]: The list of components contained in the collection.
    """
    return [ x for x in self.values() if type(x) is Component ]
    
  def add(self, d: Description):
    """
    Add a description to the collection.

    Args:
        d (Description): Description to add in the collection.
    """
    if d.name not in self:
      self[d.name] = d

  @classmethod
  def from_graph(cls, 
    graph: rdflib.Graph | str, 
    component_class: rdflib.URIRef = rdflib.OWL.Class,
    description_class: rdflib.URIRef = rdflib.OWL.Class) -> "DescriptionCollection":
    """
    Load a collection of descriptions from an RDF file.

    Args:
        graph (rdflib.Graph): Input rdf graph.
        description_classes (List[rdflib.URIRef], optional): Classes used to define descriptions. Defaults to [rdflib.OWL.Class].
        progress (bool, optional): Show progress when parsing the graph. Defaults to False.

    Returns:
        DescriptionCollection: Return the newly built collection of descriptions.
    """
    if type(graph) == str:
      graph = rdflib.Graph().parse(graph)

    dc = cls() # create an empty description collection

    # extract all the components from ontology
    for row in graph.query(cls.COMPONENTS_QUERY.format(component_class=component_class)):
      dc.add(Component(str(row.component)))

    # extract all the descriptions from the ontology
    descriptions_query_results = graph.query(cls.DESCRIPTIONS_QUERY.format(
      component_class=component_class,
      description_class=description_class))
    for row in descriptions_query_results:
      dc.add(Description(str(row.description)))

    # add all the components
    for row in descriptions_query_results:
      if str(row.description) in dc and str(row.element) in dc:
        dc[str(row.description)].add_element(dc[str(row.element)])

    # add the hierarchy
    for row in graph.query(cls.HIERARCHY_QUERY.format(
      component_class=component_class,
      description_class=description_class)):
      if str(row.element) in dc and str(row.parent) in dc:
        element = dc[str(row.element)]
        parent = dc[str(row.parent)]
        element.add_parent(parent)
        parent.add_child(element)

    return dc
