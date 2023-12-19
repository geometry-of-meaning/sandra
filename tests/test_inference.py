import sandra
import numpy as np

TOY_EXAMPLE_FRAME_PATH = "examples/toy_example_frame/ontology.owl"
dc = sandra.DescriptionCollection.from_graph(TOY_EXAMPLE_FRAME_PATH)
reasoner = sandra.Reasoner(dc)


def test_satisfy_specific():
  s = sandra.Situation([
    dc["https://w3id.org/geometryofmeaning/toy_example_frame/Goods"], 
    dc["https://w3id.org/geometryofmeaning/toy_example_frame/Buyer"]])
  enc_s = reasoner.encode(s)

  infer = reasoner.infer(enc_s)[0]

  # check that the first frame (Commerce_buy) has probability ~1
  assert np.allclose(infer[0], 1, rtol=0.01)

  # check that the similar frame (Importing) has probability < than Commerce_buy
  assert not np.allclose(infer[2], 1, rtol=0.01)
  assert infer[2] < infer[0]

def test_satisfy_using_a_composed_situation():
  s = sandra.Situation([
    sandra.Situation([dc["https://w3id.org/geometryofmeaning/toy_example_frame/Quantity"]]),
    dc["https://w3id.org/geometryofmeaning/toy_example_frame/Buyer"]])
  enc_s = reasoner.encode(s)

  infer = reasoner.infer(enc_s)[0]

  # check that the first frame (Commerce_buy) has probability ~1
  assert np.allclose(infer[0], 1, rtol=0.01)

  # check that the similar frame (Importing) has probability < than Commerce_buy
  assert not np.allclose(infer[2], 1, rtol=0.01)
  assert infer[2] < infer[0]

def test_satisfy_using_a_superclass():
  s = sandra.Situation([
    dc["https://w3id.org/geometryofmeaning/toy_example_frame/Goods"], 
    dc["https://w3id.org/geometryofmeaning/toy_example_frame/Agent"]])
  enc_s = reasoner.encode(s)

  infer = reasoner.infer(enc_s)[0]
  
  # check that both frames involving Agent and Goods are
  # satisfied 0.01 < p < 0.9
  assert infer[0] > 0.01 and infer[0] < 0.9
  assert infer[2] > 0.01 and infer[2] < 0.9

def test_satisfy_using_both_superclass():
  s = sandra.Situation([
    dc["https://w3id.org/geometryofmeaning/toy_example_frame/Asset"], 
    dc["https://w3id.org/geometryofmeaning/toy_example_frame/Agent"]])
  enc_s = reasoner.encode(s)

  infer = reasoner.infer(enc_s)[0]
  assert infer[0] > 0.01 and infer[0] < 0.9
  assert infer[2] > 0.01 and infer[2] < 0.9

def test_satisfy_goods():
  s = sandra.Situation([dc["https://w3id.org/geometryofmeaning/toy_example_frame/Quantity"]])
  enc_s = reasoner.encode(s)

  infer = reasoner.infer(enc_s)[0]
  assert infer[0] > 0.01 and infer[0] < 0.9
  assert infer[2] > 0.01 and infer[2] < 0.9