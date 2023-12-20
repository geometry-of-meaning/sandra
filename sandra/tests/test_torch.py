import sandra
from sandra.torch import ReasonerModule
import torch

TOY_EXAMPLE_FRAME_PATH = "examples/toy_example_frame/ontology.owl"
dc = sandra.DescriptionCollection.from_graph(TOY_EXAMPLE_FRAME_PATH)
reasoner = ReasonerModule(dc)

def test_components_encoding():
  # compute the encoding for all the components
  encoded = torch.stack([reasoner.encode_element(e) for e in dc.elements])
  
  # since they form a basis, they should all be independent
  # we can check this by computing the eigenvalues on the matrix
  # where the basis' vectors are arranged in the columns
  # and checking that all eigenvalues are != 0
  eigenvalues, _ =  torch.linalg.eig(encoded.T)
  assert (eigenvalues != 0).all()

def test_basis_independence():
  # check that the basis of the vector space is made by linearly independent
  # vectors by checking that all the eigenvalues are != 0
  eigenvalues, _ =  torch.linalg.eig(reasoner.basis)
  assert (eigenvalues != 0).all()

def test_satisfy_specific():
  s = sandra.Situation([
    dc["https://w3id.org/geometryofmeaning/toy_example_frame/Goods"], 
    dc["https://w3id.org/geometryofmeaning/toy_example_frame/Buyer"]])
  enc_s = reasoner.encode(s)

  infer = reasoner(enc_s)[0]

  # check that the first frame (Commerce_buy) has probability ~1
  assert torch.allclose(infer[0], torch.tensor([1]).float(), rtol=0.01)

  # check that the similar frame (Importing) has probability < than Commerce_buy
  assert not torch.allclose(infer[2], torch.tensor([1]).float(), rtol=0.01)
  assert infer[2] < infer[0]

def test_satisfy_using_a_composed_situation():
  s = sandra.Situation([
    sandra.Situation([dc["https://w3id.org/geometryofmeaning/toy_example_frame/Quantity"]]),
    dc["https://w3id.org/geometryofmeaning/toy_example_frame/Buyer"]])
  enc_s = reasoner.encode(s)

  infer = reasoner(enc_s)[0]

  # check that the first frame (Commerce_buy) has probability ~1
  assert torch.allclose(infer[0], torch.tensor([1]).float(), rtol=0.01)

  # check that the similar frame (Importing) has probability < than Commerce_buy
  assert not torch.allclose(infer[2], torch.tensor([1]).float(), rtol=0.01)
  assert infer[2] < infer[0]

def test_satisfy_using_a_superclass():
  s = sandra.Situation([
    dc["https://w3id.org/geometryofmeaning/toy_example_frame/Goods"], 
    dc["https://w3id.org/geometryofmeaning/toy_example_frame/Agent"]])
  enc_s = reasoner.encode(s)

  infer = reasoner(enc_s)[0]
  
  # check that both frames involving Agent and Goods are
  # satisfied 0.01 < p < 0.9
  assert not torch.allclose(infer[0], torch.tensor([1]).float(), rtol=0.1)
  assert not torch.allclose(infer[2], torch.tensor([1]).float(), rtol=0.1)

def test_satisfy_using_both_superclass():
  s = sandra.Situation([
    dc["https://w3id.org/geometryofmeaning/toy_example_frame/Asset"], 
    dc["https://w3id.org/geometryofmeaning/toy_example_frame/Agent"]])
  enc_s = reasoner.encode(s)

  infer = reasoner(enc_s)[0]
  assert not torch.allclose(infer[0], torch.tensor([1]).float(), rtol=0.1)
  assert not torch.allclose(infer[2], torch.tensor([1]).float(), rtol=0.1)

def test_satisfy_goods():
  s = sandra.Situation([dc["https://w3id.org/geometryofmeaning/toy_example_frame/Quantity"]])
  enc_s = reasoner.encode(s)

  infer = reasoner(enc_s)[0]
  assert infer[0] > 0.01 and infer[0] < 0.9
  assert infer[2] > 0.01 and infer[2] < 0.9

def test_satisfy_using_subclass():
  s = sandra.Situation([
    dc["https://w3id.org/geometryofmeaning/toy_example_frame/Goods"], 
    dc["https://w3id.org/geometryofmeaning/toy_example_frame/Wholesale_buyer"]])
  enc_s = reasoner.encode(s)

  infer = reasoner(enc_s)[0]
  assert torch.allclose(infer[0], torch.tensor([1]).float(), rtol=0.01)