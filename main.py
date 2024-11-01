'''
Authors:    Ragnar, Danielle and Huldar
Date:       2024-10-08
Project:
Neural network that solves the kaggle used car prices problem:
https://www.kaggle.com/competitions/playground-series-s4e9/data

Solves using a single layer neural network, with regression and a basis function as per chapter 4 in Bishop 
A special interest on missing data

Notes from Jón 2024-10-30
Hafa percent error fyrir cost function - leita að custom cost function fyrir
torch til að geta notað autograd og þurfa ekki að diffra fallið sjálf
'''
# Start timer import
import time # For measuring runtime
# Start timer
start_time = time.time()

# Start message
print(" \r\nRunning main.py")

# Settings
#--------------------------------------------------------
data_file_name = "train.csv"    # Which file to get the data from
train_ratio = 0.75               # How high ratio of data should be used for training
M = 24                          # Number of hidden nodes - 12 dimensional data. - Decided via trial and error or heuristics according to Jón but usually more than number of data dimensions in order to not compress data.
training_cycles = 20             # A.k.a "epochs" or how many times the training goes through each data point in the training data
learning_rate = 0.01           # The learning rate for the neural network training
test_eval_method = "percent"    # Which evaluation method for the error is used for testing. See tools.test_on_data for options
save_initial_weights = True            # If the weights should be saved
initial_weights_filename = "weights_initial.csv" # Filename (including path) for the weights to be saved to or read from - NOT YET IMPLEMENTED
save_final_weights = True            # If the weights should be saved
final_weights_filename = "weights_final.csv" # Filename (including path) for the weights to be saved to or read from - NOT YET IMPLEMENTED
load_weights = False            # If the weights should be loaded from initial_weights_filename - NOT YET IMPLEMENTED
should_train    = True          # If the neural network should train or just test
#--------------------------------------------------------

# Imports
print("Importing modules.............",end="")

import matplotlib.pyplot
import torch # For working with tensors and neural networks
import tools # Group tools
from colorama import Fore, Style  # For coloring terminal messages
print(Fore.GREEN + "Complete" + Style.RESET_ALL)

# Load data and transform data into manageable form
print("Loading data..................",end="")
data, targets = tools.read_in_file(data_file_name)
N, D = data.size()  # Get number of data points, N, and number of data dimensions, D.
print(Fore.GREEN + "Complete" + Style.RESET_ALL)

# Split data into training_data, training_targets, test_data & test_targets
print("Splitting data................",end="")
(train_data, train_targets), (test_data, test_targets) = tools.split_data(data,targets, train_ratio=train_ratio)
print(Fore.GREEN + "Complete" + Style.RESET_ALL)

# Initialize neural network with random weights or saved ones
# Based on https://pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html
print("Initializing neural network...",end="")
if load_weights:
    neural_network = tools.NeuralNetwork(D, M, load_weights_file=initial_weights_filename)
else:
    neural_network = tools.NeuralNetwork(D, M).to(tools.get_device())
print(Fore.GREEN + "Complete" + Style.RESET_ALL)

# Save initial weights if save weights
if save_initial_weights:
    print("Saving initial weights........",end="")
    neural_network.save_weights(initial_weights_filename)
    print(Fore.GREEN + "Complete" + Style.RESET_ALL)

# Test initial weights on test set, log errors
initial_test_time = time.time()
print("1st testing neural network....",end="")
first_testing_loss = neural_network.test_on_data(test_data, test_targets,eval_method=test_eval_method, msg="1st testing neural network....")
print(Fore.GREEN + "Complete" + Style.RESET_ALL)
initial_test_time = time.time() - initial_test_time

# Train neural network on training set
if should_train:
    train_time = time.time()
    print("Training neural network.......",end="")
    training_loss, n_unclean_points = neural_network.train_on_data(train_data, train_targets,epochs=training_cycles,lr=learning_rate)
    print(Fore.GREEN + "Complete" + Style.RESET_ALL)
    train_time = time.time() - train_time
    print("Unclean points: " + str(n_unclean_points))

# Save final weights if save weights
if save_final_weights:
    print("Saving final weights..........",end="")
    neural_network.save_weights(final_weights_filename)
    print(Fore.GREEN + "Complete" + Style.RESET_ALL)

print("Estimated car price of first car (a.k.a. data point): " + str(int(neural_network(train_data[0]).item())))
print("Supposed to be: " + str(int(train_targets[0].item())))

# Test neural network on test set, log errors
test_time = time.time()
print("Testing neural network........",end="")
testing_loss = neural_network.test_on_data(test_data, test_targets,eval_method=test_eval_method)
print(Fore.GREEN + "Complete" + Style.RESET_ALL)
test_time = time.time() - test_time

# Plot results
print("Plotting results..............",end="")
tools.plot_results(100*first_testing_loss,filename="Fig_initial_test_loss", ylabel="% error",title="Before training")
tools.plot_results(100*training_loss,filename="Fig_Training_loss", ylabel="% error",title="Training loss")
tools.plot_results(100*testing_loss,filename="Fig_after_train_test_loss", ylabel="% error",title="After training")
print(Fore.GREEN + "Complete" + Style.RESET_ALL)

# For when the above code is already working, implement the following:
#-----------------------------------------
# Train with complete data, test with complete data
# Train with complete data, test with incomplete data
# Train with incomplete data, test with complete data
# Train with incomplete data, test with incomplete data

#-----------------------------------------------

# Run final messages and print times
print("\nTraining took {:0.1f} minutes with a train ratio of {:0.1f} %".format(train_time/60, 100*train_ratio))
print("Initial testing took {:0.1f} seconds".format(initial_test_time))
print("Testing took {:0.1f} seconds".format(test_time))
print("Max error: {:.2} %".format(100*torch.max(testing_loss).item()))

# Calculate runtime and print succesful run message
runtime = time.time() - start_time
print("\nmain.py ran succesfully in " + Fore.BLUE + " {:0.1f} minutes\n".format(runtime/60)  + Style.RESET_ALL)