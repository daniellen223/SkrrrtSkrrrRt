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
import torch
import tools # Group tools

# Load data
print("Loading data..............",end="")
data, targets = tools.read_in_file()
print("Complete")

# Transform data into manageable form



# Split data into training_data, training_targets, test_data & test_targets
print("Splitting data............",end="")
(train_data, train_targets), (test_data, test_targets) = tools.split_data(data,targets, train_ratio=0.1)
print("Complete")

# Initialize neural network
# torch.nn or similar might be useful for this - Huldar 2024-10-20

# Train neural network on training set
#print("Training neural network...",end="")

#print("Complete")


# Test neural network on test set, log errors
#print("Testing neural network....",end="")

#print("Complete")

# Plot results
#print("Plotting results..........",end="")

#print("Complete")


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