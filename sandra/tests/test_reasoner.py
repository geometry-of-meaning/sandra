import sandra
import numpy as np

TOY_EXAMPLE_FRAME_PATH = "examples/toy_example_frame/ontology.owl"
ONTOLOGY_WITH_LOOPS_PATH = "examples/ontology_with_loops.ttl"

def test_components_encoding():
  dc = sandra.Ontology.from_graph(TOY_EXAMPLE_FRAME_PATH)
  reasoner = sandra.Reasoner(dc)
  # compute the encoding for all the components
  encoded = np.stack([reasoner.encode(e) for e in dc.roles])
  
  # since they form a basis, they should all be independent
  # we can check this by computing the eigenvalues on the matrix
  # where the basis' vectors are arranged in the columns
  # and checking that all eigenvalues are != 0
  eigenvalues, _ =  np.linalg.eig(encoded.T)
  assert (eigenvalues != 0).all()

def test_basis_independence():
  dc = sandra.Ontology.from_graph(TOY_EXAMPLE_FRAME_PATH)
  reasoner = sandra.Reasoner(dc)
  
  # check that the basis of the vector space is made by linearly independent
  # vectors by checking that all the eigenvalues are != 0
  eigenvalues, _ =  np.linalg.eig(reasoner.basis)
  assert (eigenvalues != 0).all()

def test_load_reasoner_with_loop():
  o = sandra.Ontology.from_graph(ONTOLOGY_WITH_LOOPS_PATH)
  reasoner = sandra.Reasoner(o)