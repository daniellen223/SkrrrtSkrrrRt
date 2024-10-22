'''
Authors:    Ragnar, Danielle and Huldar
Date:       2024-10-08
Project:
Neural network that solves the kaggle used car prices problem:
https://www.kaggle.com/competitions/playground-series-s4e9/data

Solves using a single layer neural network, with regression and a basis function as per chapter 4 in Bishop 
A special interest on missing data
'''
# Start timer import
import time # For measuring runtime
# Start timer
start_time = time.time()

# Start message
print(" \r\nRunning main.py")

# Imports
import torch # For working with tensors and neural networks
import tools # Group tools
from colorama import Fore, Back, Style  # For coloring terminal messages

# Load data and transform data into manageable form
print("Loading data..................",end="")
data, targets = tools.read_in_file()
N, D = data.size()  # Get number of data points, N, and number of data dimensions, D.
print(Fore.GREEN + "Complete" + Style.RESET_ALL)

# Split data into training_data, training_targets, test_data & test_targets
print("Splitting data................",end="")
(train_data, train_targets), (test_data, test_targets) = tools.split_data(data,targets, train_ratio=0.1)
print(Fore.GREEN + "Complete" + Style.RESET_ALL)

# Initialize neural network
# Based on https://pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html
print("Initializing neural network...",end="")
M = D # Number of hidden nodes
neural_network = tools.NeuralNetwork(D, M).to(tools.get_device())
weights = tools.init_weights(D, D)
print(Fore.GREEN + "Complete" + Style.RESET_ALL)

print("Estimated car price of first car (a.k.a. data point): " + str(neural_network(train_data[0,:])))
print("Supposed to be: " + str(train_targets[0]))

# Train neural network on training set
print("Training neural network.......",end="")
neural_network.train(train_data, train_targets)
#print(Fore.GREEN + "Complete" + Style.RESET_ALL)


# Test neural network on test set, log errors
#print("Testing neural network........",end="")

#print(Fore.GREEN + "Complete" + Style.RESET_ALL)

# Plot results
#print("Plotting results..............",end="")

#print(Fore.GREEN + "Complete" + Style.RESET_ALL)


# For when the above code is already working, implement the following:
#-----------------------------------------
# Train with complete data, test with complete data
# Train with complete data, test with incomplete data
# Train with incomplete data, test with complete data
# Train with incomplete data, test with incomplete data

# Testing area
#-----------------------------------------------


# Calculate runtime
runtime = time.time() - start_time

# Run finished message
print("\nmain.py ran succesfully in {:0.1f} seconds\n".format(runtime))