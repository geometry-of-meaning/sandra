import sandra

TOY_EXAMPLE_FRAME_PATH = "examples/toy_example_frame/ontology.owl"

def test_load_graph_from_filename():
  sandra.DescriptionCollection.from_graph(TOY_EXAMPLE_FRAME_PATH)

def test_load_graph_from_rdflib_graph():
  import rdflib
  graph = rdflib.Graph().parse(TOY_EXAMPLE_FRAME_PATH)
  sandra.DescriptionCollection.from_graph(graph)

def test_load_components():
  dc = sandra.DescriptionCollection.from_graph(TOY_EXAMPLE_FRAME_PATH)
  components = [x for x in dc.values() if type(x) is sandra.Component]
  assert len(components) == 16
  assert len(dc.components) == 16
  assert components == dc.components

def test_load_components_with_class_name():
  import rdflib
  
  # Components are asserted using owl.Class as top-class, not RDFS.Class
  dc = sandra.DescriptionCollection.from_graph(
    TOY_EXAMPLE_FRAME_PATH, 
    component_class=rdflib.RDFS.Class)
  components = [x for x in dc.values() if type(x) is sandra.Component]
  assert len(components) == 0

def test_load_descriptions():
  dc = sandra.DescriptionCollection.from_graph(TOY_EXAMPLE_FRAME_PATH)
  descriptions = [x for x in dc.values() if type(x) is sandra.Description]
  assert len(descriptions) == 4

def test_load_descriptions_with_class_name():
  import rdflib
  # Descriptions are asserted using owl.Class as top-class, not RDFS.Class
  dc = sandra.DescriptionCollection.from_graph(
    TOY_EXAMPLE_FRAME_PATH, 
    description_class=rdflib.RDFS.Class)
  descriptions = [x for x in dc.values() if type(x) is sandra.Description]
  assert len(descriptions) == 0

  dc = sandra.DescriptionCollection.from_graph(
    TOY_EXAMPLE_FRAME_PATH, 
    description_class=rdflib.RDFS.Class)
  descriptions = [x for x in dc.values() if type(x) is sandra.Description]
  assert len(descriptions) == 0

def test_load_descriptions_elements():
  dc = sandra.DescriptionCollection.from_graph(TOY_EXAMPLE_FRAME_PATH)
  description = dc["https://w3id.org/geometryofmeaning/toy_example_frame/Commerce_buy"]
  assert len(description.elements) == 2

def test_load_hierarchy():
  dc = sandra.DescriptionCollection.from_graph(TOY_EXAMPLE_FRAME_PATH)
  quality = dc["https://w3id.org/geometryofmeaning/toy_example_frame/Quality"]
  quantity = dc["https://w3id.org/geometryofmeaning/toy_example_frame/Quantity"]
  assert quantity in quality.children
  assert quality in quantity.parents
