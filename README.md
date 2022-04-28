# DbPedia-contents-classification
Classifying Contents in DbPedia dataset using a VGG-based network

This repo is related to the classification of DBpedia sentences. DbPedia dataset is a large free access dataset of ontology contents which comprise up more than 630k in 14 categories which namely are: Company, EducationalInstitution, Artist, Athlete, OfficeHolder, MeanOfTransportation, Building, NaturalPlace, Village, Animal, Plant, Album, Film, WrittenWork. 

This dataset has been already split into training and test sets. To classify DbPedia contexts, a new VGG-based CNN model has been designed. Since this model is a CNN, we considered each sentence as 1014 characters and converted each sentence to a 1014 ⨉ 16 matrix, leveraging an embedding layer in Tensorflow library. This model managed to classify the test set samples with the accuracy of 94.25 %. This model contains one embedding layer, 9 convolution layers, and three fully-connected layers. The below fig. shows the model architecture:

![image](https://user-images.githubusercontent.com/41271921/162592784-21d3d556-4012-41cd-92ff-e691ac3349bd.png)
 

## Requirements
Python > 3

Tensorflow == 1.15 

NLTK
