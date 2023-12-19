# sandra
Situation And Description Reasoner based on linear Algebra

## Example

```python
>>> import sandra
>>> dc = sandra.DescriptionCollection.from_graph("examples/toy_example_frame/ontology.owl")
>>> reasoner = sandra.reasoner.Reasoner(dc)
>>> # Situation defined using specific elements
>>> # satisfies only a specific description: Commerce_buy
>>> s = sandra.Situation([
>>>   dc["https://w3id.org/geometryofmeaning/toy_example_frame/Goods"], 
>>>   dc["https://w3id.org/geometryofmeaning/toy_example_frame/Buyer"]])
>>> for desc, score in reasoner.classify(s):
>>>   print(f"{score:1.3f} - {desc.name}")
1.000 - https://w3id.org/geometryofmeaning/toy_example_frame/Commerce_buy
1.000 - https://w3id.org/geometryofmeaning/toy_example_frame/Goods
0.250 - https://w3id.org/geometryofmeaning/toy_example_frame/Importing
0.000 - https://w3id.org/geometryofmeaning/toy_example_frame/Motion
>>> # Situation defined using more general elements
>>> # since a Buyer -> Agent but the inverse is not true, the description
>>> # is not fully satisfied. It does satisfy a little the Importing
>>> # one, since it is composed of Asset and an Importer (which is an agent)
>>> s = sandra.Situation([
>>>   dc["https://w3id.org/geometryofmeaning/toy_example_frame/Asset"], 
>>>   dc["https://w3id.org/geometryofmeaning/toy_example_frame/Agent"]])
>>> for desc, score in reasoner.classify(s):
>>>   print(f"{score:1.3f} - {desc.name}")
0.500 - https://w3id.org/geometryofmeaning/toy_example_frame/Commerce_buy
0.000 - https://w3id.org/geometryofmeaning/toy_example_frame/Goods
0.250 - https://w3id.org/geometryofmeaning/toy_example_frame/Importing
0.000 - https://w3id.org/geometryofmeaning/toy_example_frame/Motion
>>> # Situation defined by nesting different situations
>>> # Goods is a description that requires a Quantity to be asserted
>>> # hence the situation involving quantity satisfies the Goods one,
>>> # which completes the satisfiability of Commerce_buy
>>> s = sandra.Situation([
>>>   sandra.Situation([dc["https://w3id.org/geometryofmeaning/toy_example_frame/Quantity"]]), 
>>>   dc["https://w3id.org/geometryofmeaning/toy_example_frame/Buyer"]])
>>> for desc, score in reasoner.classify(s):
>>>   print(f"{score:1.3f} - {desc.name}")  
1.000 - https://w3id.org/geometryofmeaning/toy_example_frame/Commerce_buy
1.000 - https://w3id.org/geometryofmeaning/toy_example_frame/Goods
0.250 - https://w3id.org/geometryofmeaning/toy_example_frame/Importing
0.000 - https://w3id.org/geometryofmeaning/toy_example_frame/Motion
>>> # Only satisying a component is not enough for any description
>>> # even though Commerce_buy is defined using Goods it is not fully satisfied
>>> s = sandra.Situation([dc["https://w3id.org/geometryofmeaning/toy_example_frame/Quantity"]])
>>> for desc, score in reasoner.classify(s):
>>>   print(f"{score:1.3f} - {desc.name}")
0.500 - https://w3id.org/geometryofmeaning/toy_example_frame/Commerce_buy
1.000 - https://w3id.org/geometryofmeaning/toy_example_frame/Goods
0.250 - https://w3id.org/geometryofmeaning/toy_example_frame/Importing
0.000 - https://w3id.org/geometryofmeaning/toy_example_frame/Motion
```