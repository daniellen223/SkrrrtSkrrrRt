'''
Authors:    Ragnar, Danielle and Huldar
Date:       2024-10-08
Project:
Neural network that solves the kaggle used car prices problem:
https://www.kaggle.com/competitions/playground-series-s4e9/data

Solves using a single layer neural network, with regression and a basis function as per chapter 4 in Bishop 
A special interest on missing data
'''
print("\nRunning main.py")
# Imports
import torch
import tools # Group tools

# Load data
#print("Loading data")



#print("Complete\n")

# Transform data into manageable form



# Split data into training_data, training_targets, test_data & test_targets
#print("Splitting data")



#print("Complete\n")

# Initialize neural network


# Train neural network on training set
#print("Training neural network")



#print("Complete\n")


# Test neural network on test set, log errors
#print("Testing neural network")



#print("Complete\n")

# Plot results
#print("Plotting results")



#print("Complete\n")


# For when the above code is already working, implement the following:
#-----------------------------------------
# Train with complete data, test with complete data
# Train with complete data, test with incomplete data
# Train with incomplete data, test with complete data
# Train with incomplete data, test with incomplete data

# Testing area
#-----------------------------------------------
x = torch.ones(3)
w = torch.rand(3)
w = w-0.5
print("activation function test")
print("w: " + str(w))
print(tools.h(x,w))

# Run finished message
print("\n")
print("main.py ran succesfully")
