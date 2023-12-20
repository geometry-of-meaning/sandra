import sandra
import numpy as np

TOY_EXAMPLE_FRAME_PATH = "examples/toy_example_frame/ontology.owl"
dc = sandra.DescriptionCollection.from_graph(TOY_EXAMPLE_FRAME_PATH)
reasoner = sandra.Reasoner(dc)

def test_components_encoding():
  # compute the encoding for all the components
  encoded = np.stack([reasoner.encode_element(e) for e in dc.elements])
  
  # since they form a basis, they should all be independent
  # we can check this by computing the eigenvalues on the matrix
  # where the basis' vectors are arranged in the columns
  # and checking that all eigenvalues are != 0
  eigenvalues, _ =  np.linalg.eig(encoded.T)
  assert (eigenvalues != 0).all()

def test_basis_independence():
  # check that the basis of the vector space is made by linearly independent
  # vectors by checking that all the eigenvalues are != 0
  eigenvalues, _ =  np.linalg.eig(reasoner.basis)
  assert (eigenvalues != 0).all()
