import sandra

TOY_EXAMPLE_FRAME_PATH = "examples/toy_example_frame/ontology.owl"
ONTOLOGY_WITH_LOOPS_PATH = "examples/ontology_with_loops.ttl"

def test_load_graph_from_filename():
  sandra.Ontology.from_graph(TOY_EXAMPLE_FRAME_PATH)

def test_load_graph_from_rdflib_graph():
  import rdflib
  graph = rdflib.Graph().parse(TOY_EXAMPLE_FRAME_PATH)
  sandra.Ontology.from_graph(graph)

def test_load_roles():
  o = sandra.Ontology.from_graph(TOY_EXAMPLE_FRAME_PATH)
  roles = [x for x in o.values() if x.is_role]
  assert len(roles) == 17
  assert len(o.roles) == 17
  assert roles == o.roles

def test_load_components_with_class_name():
  import rdflib
  
  # Roles are asserted using owl.Class as top-class, not RDFS.Class
  o = sandra.Ontology.from_graph(
    TOY_EXAMPLE_FRAME_PATH, 
    role_class=rdflib.RDFS.Class,
    description_class=rdflib.RDFS.Class)
  assert len(o.roles) == 0
  assert len(o.descriptions) == 0

def test_load_descriptions():
  o = sandra.Ontology.from_graph(TOY_EXAMPLE_FRAME_PATH)
  assert len(o.descriptions) == 4

def test_load_descriptions_with_class_name():
  import rdflib
  # Descriptions are asserted using owl.Class as top-class, not RDFS.Class
  o = sandra.Ontology.from_graph(TOY_EXAMPLE_FRAME_PATH, description_class=rdflib.RDFS.Class)
  assert len(o.descriptions) == 0

  o = sandra.Ontology.from_graph(TOY_EXAMPLE_FRAME_PATH, description_class=rdflib.RDFS.Class)
  assert len(o.descriptions) == 0

def test_load_descriptions_elements():
  o = sandra.Ontology.from_graph(TOY_EXAMPLE_FRAME_PATH)
  description = o["https://w3id.org/geometryofmeaning/toy_example_frame/Commerce_buy"]
  assert len(description.components) == 2

def test_load_hierarchy():
  o = sandra.Ontology.from_graph(TOY_EXAMPLE_FRAME_PATH)
  quality = o["https://w3id.org/geometryofmeaning/toy_example_frame/Quality"]
  quantity = o["https://w3id.org/geometryofmeaning/toy_example_frame/Quantity"]
  assert quantity in quality.children
  assert quality in quantity.parents

def test_error_direct_recursion():
  o = sandra.Ontology()
  d = sandra.Element("aa")
  d.add(d)
  assert len(d.components) == 0

def test_adding_element_twice():
  o = sandra.Ontology()
  o.add(sandra.Element("aa"))
  o.add(sandra.Element("aa"))
  assert len(o.elements) == 1
