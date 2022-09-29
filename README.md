# Autism Classification

![Autism Classification Macro Architecture][macro_architecture]

The subject repository is responsible for Autism Classification. The subject architecture is trained autistic children dataset. Currently inference architecture supports two modes for classification.

* Classification using classification layer
* Classification using one shot learning

In terms of classification using classification layer, the pretrained model can classify two follwoing classes.

* Autism
* Non-Autism

In terms of classification using one shot learning, the pretrained model classifies the sample on the basis of base data similarity.

The model used in this architecture is ***ResNet-50***, trained on subject data using ***Pytorch*** framework.

##### For extensive documentation, please check [***wiki***](https://github.com/codeadeel/Autism-Classification/wiki).

[macro_architecture]: ./MarkDown-Data/macro_architecture.jpg