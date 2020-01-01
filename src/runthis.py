import mnist_loader
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

#1----------------------------------using quadratic cost function---------------------------------------

#import network
#net = network.Network([784, 10, 10])
#net.SGD(training_data, 30, 10, 3.0, test_data=test_data)

#2-----------------------------using cross-entropy cost---------------------------------------------------

import network2
net = network2.Network([784, 100, 10], cost=network2.CrossEntropyCost)
net.large_weight_initializer()  #random gaussian weights initializer with mean 0 and variance 1---- #--comment this if you use #9

#3-------------------------------------------------------------------------------------------------------

#net.SGD(training_data, 30, 10, 0.5, evaluation_data=test_data, monitor_evaluation_accuracy=True)

#4------------------------demonstration of overfitting with less data-----------------------------------------------

#net.SGD(training_data[:1000], 400, 10, 1, evaluation_data=test_data, monitor_training_accuracy=True, monitor_evaluation_accuracy=True, monitor_training_cost=True, monitor_evaluation_cost=True)

#5------------------------with regularization-----------------------------------------------------------------------

net.SGD(training_data[:1000], 400, 10, 1, evaluation_data=test_data, lmbda = 0.1, monitor_evaluation_cost=True, monitor_evaluation_accuracy=True, monitor_training_cost=True, monitor_training_accuracy=True)

#6------------------------full data with regularization-----------------------------------------------------------

#net.SGD(training_data, 30, 10, 0.5, evaluation_data=test_data, lmbda = 5.0, monitor_evaluation_accuracy=True, monitor_training_accuracy=True)


#7------------------------full data with regularization with more neurons-------change NN to 100----------------------------------

#net.SGD(training_data, 30, 10, 0.5, lmbda=5.0, evaluation_data=validation_data, monitor_evaluation_cost=True, monitor_evaluation_accuracy=True)

#8------------------------ 98% with below on Validation data!------------------------------------------------------

#net.SGD(training_data, 60, 10, 0.1, lmbda=5.0, evaluation_data=validation_data, monitor_evaluation_accuracy=True)


#9------------------------with (1/sqrt(n_in)) weight initialization to reduce variance in weights and hence in Z (weighted o/p)

#net.SGD(training_data, 30, 10, 0.1, lmbda = 5.0, evaluation_data=validation_data, monitor_evaluation_accuracy=True)

#10-------------------with all monitoring flags on-------------------

#evaluation_cost, evaluation_accuracy, training_cost, training_accuracy = net.SGD(training_data, 30, 10, 0.5, lmbda =5.0, evaluation_data=validation_data, monitor_evaluation_accuracy=True, monitor_evaluation_cost=True, monitor_training_accuracy=True, monitor_training_cost=True)
#net.save('./saved_NN')                                   #-------- save the network

#--------------------------convolutional neuralnet------#dont run below code until you have set theano properly!------------------------------

#import network3
#from network3 import Network
#from network3 import ConvPoolLayer, FullyConnectedLayer, SoftmaxLayer
#training_data, validation_data, test_data = network3.load_data_shared()
#mini_batch_size = 10

#11-------------------------------------------------------------------------------------------------------------

#net = Network([FullyConnectedLayer(n_in=784, n_out=100), SoftmaxLayer(n_in=100, n_out=10)], mini_batch_size)
#net.SGD(training_data, 60, mini_batch_size, 0.1, validation_data, test_data)

#12---------------------------------------------------------------------------------------------------------

#net = Network([
#    ConvPoolLayer(image_shape=(mini_batch_size, 1, 28, 28), 
#                  filter_shape=(20, 1, 5, 5), 
#                  poolsize=(2, 2)),
#    FullyConnectedLayer(n_in=20*12*12, n_out=100),
#    SoftmaxLayer(n_in=100, n_out=10)], mini_batch_size)
#net.SGD(training_data, 60, mini_batch_size, 0.1, 
#            validation_data, test_data) 

#13----------------------------------------two conv layers-----------------------------------------------------

#net = Network([
#    ConvPoolLayer(image_shape=(mini_batch_size, 1, 28, 28), 
#                  filter_shape=(20, 1, 5, 5),                 
#                  poolsize=(2, 2)),                         
#    ConvPoolLayer(image_shape=(mini_batch_size, 20, 12, 12),
#                  filter_shape=(40, 20, 5, 5),              
#                  poolsize=(2, 2)),                         
#    FullyConnectedLayer(n_in=40*4*4, n_out=100),             
#    SoftmaxLayer(n_in=100, n_out=10)], mini_batch_size)     
#net.SGD(training_data, 60, mini_batch_size, 0.1,            
#        validation_data, test_data)

#14-----------------------------------------with ReLu and L2 regularisation-----------------------------------------------------
#from network3 import ReLU
#net = Network([
#    ConvPoolLayer(image_shape=(mini_batch_size, 1, 28, 28),
#                  filter_shape=(20, 1, 5, 5),
#                  poolsize=(2, 2),
#                  activation_fn=ReLU),
#    ConvPoolLayer(image_shape=(mini_batch_size, 20, 12, 12),
#                  filter_shape=(40, 20, 5, 5),
#                  poolsize=(2, 2),
#                  activation_fn=ReLU),
#    FullyConnectedLayer(n_in=40*4*4, n_out=100, activation_fn=ReLU),
#    SoftmaxLayer(n_in=100, n_out=10)], mini_batch_size)
#net.SGD(training_data, 60, mini_batch_size, 0.03,
#        validation_data, test_data, lmbda=0.1)

#15------------------------------------with expanded (translated by a pixel) data set---------------------------------------

# python3 expand_mnist.py ---------run this before runniing the below code

#expanded_training_data, _, _ = network3.load_data_shared(            #---uncomment for #17---
#    "../data/mnist_expanded.pkl.gz")
#net = Network([
#    ConvPoolLayer(image_shape=(mini_batch_size, 1, 28, 28),
#                  filter_shape=(20, 1, 5, 5),
#                  poolsize=(2, 2),
#                  activation_fn=ReLU),
#    ConvPoolLayer(image_shape=(mini_batch_size, 20, 12, 12),
#                  filter_shape=(40, 20, 5, 5),
#                  poolsize=(2, 2),
#                  activation_fn=ReLU),
#    FullyConnectedLayer(n_in=40*4*4, n_out=100, activation_fn=ReLU),
#    SoftmaxLayer(n_in=100, n_out=10)], mini_batch_size)
#net.SGD(expanded_training_data, 60, mini_batch_size, 0.03,
#        validation_data, test_data, lmbda=0.1)

#16--------------------------------add another FC 100 NN------------------------------------

#net = Network([
#    ConvPoolLayer(image_shape=(mini_batch_size, 1, 28, 28),
#                  filter_shape=(20, 1, 5, 5),
#                  poolsize=(2, 2),
#                  activation_fn=ReLU),
#    ConvPoolLayer(image_shape=(mini_batch_size, 20, 12, 12),
#                  filter_shape=(40, 20, 5, 5),
#                  poolsize=(2, 2),
#                  activation_fn=ReLU),
#    FullyConnectedLayer(n_in=40*4*4, n_out=100, activation_fn=ReLU),
#    FullyConnectedLayer(n_in=100, n_out=100, activation_fn=ReLU),
#    SoftmaxLayer(n_in=100, n_out=10)], mini_batch_size)
#net.SGD(expanded_training_data, 60, mini_batch_size, 0.03,
#        validation_data, test_data, lmbda=0.1)

#17-Add dropout and reduce epochs to 40--and 1000 NN to compesate for dropout training (300 will be fine)---99.6!---------------

#net = Network([
#    ConvPoolLayer(image_shape=(mini_batch_size, 1, 28, 28),
#                  filter_shape=(20, 1, 5, 5),
#                  poolsize=(2, 2),
#                  activation_fn=ReLU),
#    ConvPoolLayer(image_shape=(mini_batch_size, 20, 12, 12),
#                  filter_shape=(40, 20, 5, 5),
#                  poolsize=(2, 2),
#                  activation_fn=ReLU),
#    FullyConnectedLayer(
#        n_in=40*4*4, n_out=1000, activation_fn=ReLU, p_dropout=0.5),
#    FullyConnectedLayer(
#        n_in=1000, n_out=1000, activation_fn=ReLU, p_dropout=0.5),
#    SoftmaxLayer(n_in=1000, n_out=10, p_dropout=0.5)],
#    mini_batch_size)
#net.SGD(expanded_training_data, 40, mini_batch_size, 0.03,
#            validation_data, test_data)
