# Neural-Network

Description
===========

Feed-forward neural network from ***scratch*** (the only libraries I'm using are java libraries, and mostly used for reading training data) using Java. The purpose of project is to classify images as digits (numbers from 0 - 9) based on the way they look to the program using deep learning, with the help of the MNIST dataset (https://yann.lecun.com/exdb/mnist/).
Features include:
- support with 28x28 pixel handwritten digits
- backpropagation using gradient descent
- Matrix class
- 96% accuracy on 10,000 testing images (accuracy varies due to randomness at the beginning of training)
- Customizable network layers, learning rates, and number of training epochs
- Images that weren't classified correctly get stored in errors.txt as images made from ASCII text
- Data from the training epochs are stored in data.txt
- Example lists of weights and biases are stored in "savedBiases" and "savedWeights" respectively

This project is not just limited to the MNIST dataset, it was just the input and training data that I used because of the abundance of data that was available. The input layer can be adjusted to suit many different types of classification, as long as the input data is a floating number between 0 and 1. The output layer can also be adjusted accordingly, along with any hidden layers that you would like to include.

The downsides of this project are all related to the structure of feed-forward neural networks. Some of these cons include overfitting, very computationally expensive, and lots of training data is required to get an accurate classification. With that said, this network gets the job done pretty well for the data that it is classifying right now, but there are definitely better alternatives out there.

How to run
==========

Make sure you are in the /Neural-Network directory, then run:

`javac Main.java`

`java Main`
