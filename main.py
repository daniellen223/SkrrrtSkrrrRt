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
print("Importing modules.............",end="")

import matplotlib.pyplot
import torch # For working with tensors and neural networks
import tools # Group tools
from colorama import Fore, Back, Style  # For coloring terminal messages
print(Fore.GREEN + "Complete" + Style.RESET_ALL)

# Load data and transform data into manageable form
print("Loading data..................",end="")
data, targets = tools.read_in_file()
N, D = data.size()  # Get number of data points, N, and number of data dimensions, D.
print(Fore.GREEN + "Complete" + Style.RESET_ALL)

# Split data into training_data, training_targets, test_data & test_targets
print("Splitting data................",end="")
train_ratio = 0.05
(train_data, train_targets), (test_data, test_targets) = tools.split_data(data,targets, train_ratio=train_ratio)
print(Fore.GREEN + "Complete" + Style.RESET_ALL)

# Initialize neural network
# Based on https://pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html
print("Initializing neural network...",end="")
M = D # Number of hidden nodes
neural_network = tools.NeuralNetwork(D, M).to(tools.get_device())
# weights = tools.init_weights(D, D) # Not used by torch so far?
print(Fore.GREEN + "Complete" + Style.RESET_ALL)

print("Estimated car price of first car (a.k.a. data point): " + str(neural_network(train_data[0]).item()))
print("Supposed to be: " + str(train_targets[0].item()))

# Train neural network on training set
train_time = time.time()
print("Training neural network.......",end="")
# Use train mode?
training_loss = neural_network.train_on_data(train_data, train_targets,epochs=2)
print(Fore.GREEN + "Complete" + Style.RESET_ALL)
train_time = time.time() - train_time

print("Estimated car price of first car (a.k.a. data point): " + str(neural_network(train_data[0]).item()))
print("Supposed to be: " + str(train_targets[0].item()))


# Test neural network on test set, log errors
test_time = time.time()
print("Testing neural network........",end="")
testing_loss = neural_network.test_on_data(test_data, test_targets)
print(Fore.GREEN + "Complete" + Style.RESET_ALL)
test_time = time.time() - test_time

# Plot results
print("Plotting results..............",end="")
tools.plot_results(testing_loss,filename="Fig1")
print(Fore.GREEN + "Complete" + Style.RESET_ALL)

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

# Run finished message and print times
print("\nmain.py ran succesfully in {:0.1f} seconds\n".format(runtime))
print("Training took {:0.1f} seconds with a train ratio of {:0.1f} %".format(train_time, 100*train_ratio))
print("Testing took {:0.1f} seconds".format(test_time))
