import mnist_loader
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

#----------------------------------using quadratic cost function---------------------------------------

#import network
#net = network.Network([784, 10, 10])
#net.SGD(training_data, 30, 10, 3.0, test_data=test_data)

#-----------------------------using cross-entropy cost---------------------------------------------------

import network2
net = network2.Network([784, 30, 10], cost=network2.CrossEntropyCost)
net.large_weight_initializer()

#-------------------------------------------------------------------------------------------------------

#net.SGD(training_data, 30, 10, 0.5, evaluation_data=test_data, monitor_evaluation_accuracy=True)

#------------------------demonstration of overfitting with less data-----------------------------------------------

#net.SGD(training_data[:1000], 400, 10, 0.5, evaluation_data=test_data, monitor_evaluation_accuracy=True, monitor_training_cost=True)

#------------------------with regularization-----------------------------------------------------------------------

net.SGD(training_data[:1000], 400, 10, 0.5, evaluation_data=test_data, lmbda = 0.1, monitor_evaluation_cost=True, monitor_evaluation_accuracy=True, monitor_training_cost=True, monitor_training_accuracy=True)

#------------------------full data with regularization-----------------------------------------------------------

#net.SGD(training_data, 30, 10, 0.5, evaluation_data=test_data, lmbda = 5.0, monitor_evaluation_accuracy=True, monitor_training_accuracy=True)


#------------------------full data with regularization with more neurons------------------------------------------

#net = network2.Network([784, 100, 10], cost=network2.CrossEntropyCost) ----Dont uncomment this, just change 30 to 100 in line 13
#net.large_weight_initializer()                                         ----Dont uncomment this as well.
#net.SGD(training_data, 30, 10, 0.5, lmbda=5.0, evaluation_data=validation_data, monitor_evaluation_accuracy=True)

#------------------------ 98% with below on Validation data!------------------------------------------------------

#net.SGD(training_data, 60, 10, 0.1, lmbda=5.0, evaluation_data=validation_data, monitor_evaluation_accuracy=True)
