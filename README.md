# Topic Modeler
Creating topic distribution from documents 

# How it Works
The model creats a matrix of counts of the words in the documents using the [CountVectorizer](), the matrix __DWC__. The matric __DWC__ is used to generate the [Pointwise Mutual Information](https://en.wikipedia.org/wiki/Pointwise_mutual_information) values then the values are passed through a [sigmoid](https://en.wikipedia.org/wiki/Sigmoid_function) to normalize between (0, 1).
