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
import datetime # For displaying the start time in a readable format
# Start timer
start_time = time.time()

# Start message
print(" \r\nRunning main.py at " + datetime.datetime.now().strftime("%H:%M:%S"))

# Settings
#--------------------------------------------------------
data_file_name = "train.csv"    # Which file to get the data from
map_data = True            # If the data should be mapped
save_mapping = True            # Saves the data mapping for the read data to a csv file
mapping_folder = "Data_mapping" # Name of mapping folder
normalize_data = False           # If the data should be normalized before splitting into training and testing
normalized_data_filename = "Data_Normalized.csv" # The filename to save the normalized data to
all_fieldnames = ['id','brand','model','model_year','milage','fuel_type','engine','transmission','ext_col','int_col','accident','clean_title', 'price'] # The names of the columns of the input data from the csv file in the same order as it appears in the csv file
used_fields = ['brand','model_year','milage','accident','clean_title'] # Which fields we plan on using for the data
shuffle_data = False             # If the data should be randomized before splitting into training and testing
load_weights = False            # If the weights should be loaded from initial_weights_filename
initial_weights_filename = "weights_initial.csv" # Filename (including path) for the weights to be saved to or read from
should_train    = True          # If the neural network should train or just test
train_ratio = 0.1               # How high ratio of data should be used for training
nodes = [5, 128, 64, 1]                # Number of hidden nodes per layer - 11 dimensional data. First layer is equal to number of data dimensions, last layer is equal to output dimensions (1 in this case) - Decided via trial and error or heuristics according to Jón but usually more than number of data dimensions in order to not compress data.
use_generated_layer_stacks = False              # If we should use the nodes in the nodes variable or generate layer stacks.
max_layer_stacks = 12   # How many layer stacks we add maximum
training_cycles = 40            # A.k.a "epochs" or how many times the training goes through each data point in the training data
learning_rate = [0.5] #[0.001,0.005,0.01,0.05,0.1,0.5,1,5,10,50,100,500,1000]              # The learning rate for the neural network training
save_initial_weights = True            # If the weights should be saved
save_final_weights = True            # If the weights should be saved
final_weights_filename = "weights_final.csv" # Filename (including path) for the weights to be saved to or read from - NOT YET IMPLEMENTED
test_eval_method = "percent"    # Which evaluation method for the error is used for testing. See tools.test_on_data for options
save_errors     = True          # If we should save the errors as a csv file
error_file      = "percent_error.csv"   # Name of error csv file
show_plots      = False         # If the plots should be showed or not. Note: Plots are always saved
#--------------------------------------------------------

# Imports
print("Importing modules.............",end="")

import matplotlib.pyplot
import torch # For working with tensors and neural networks
import tools # Group tools
from pathlib import Path # For paths of files
from colorama import Fore, Style  # For coloring terminal messages
print(Fore.GREEN + "Complete" + Style.RESET_ALL)

# Testing area
#--------------------------------------------------------

#--------------------------------------------------------

# Load data and transform data into manageable form
data, targets = tools.read_in_file(data_file_name, save_mapping=save_mapping, mapping_folder_name=mapping_folder, should_map=map_data) #, encode=encode)
N, D = data.size()  # Get number of data points, N, and number of data dimensions, D.

# Normalize data
# If normalize is True, normalize data
if normalize_data:
    data, means, stds = tools.normalize_tensor(torch.cat((data,targets.unsqueeze(1)),1))
    tools.tensor_to_csv(data, normalized_data_filename, fieldnames=all_fieldnames)
    targets = data[:][11]
    data = data[:][0:11]
    
    # targets, target_mean, target_std = tools.normalize_tensor(targets)

# Use only the specified fields
data = tools.reduce_data_by_fields(data, all_fieldnames, used_fields)

# Split data into training_data, training_targets, test_data & test_targets
(train_data, train_targets), (test_data, test_targets) = tools.split_data(data,targets, train_ratio=train_ratio, shuffle=shuffle_data)

# For each layer_stack version, initialize neural network, train, test and save the results
layer_stacks = tools.make_layer_stacks(len(used_fields), 1, max_layer_stacks)
num_layer_stacks = len(layer_stacks)

# Store mean and max errors for each layer stack for plotting
layer_stacks_mean_errors = []
layer_stacks_max_errors = []

if not(use_generated_layer_stacks):
    num_layer_stacks = 1

for layer_stack_ID in range(num_layer_stacks):
    # For each learning rate being tested
    for i in range(len(learning_rate)):
        # Update nodes in case we want to control them
        if use_generated_layer_stacks:
            nodes = layer_stacks[layer_stack_ID]

        # Make folder name from nodes
        path = "./results/" + str(layer_stacks[layer_stack_ID]) + "/"
        file = Path(path)
        file.mkdir(parents=True, exist_ok=True)

        # Initialize neural network with random weights or saved ones
        if load_weights:
            neural_network = tools.NeuralNetwork(nodes, load_weights_file=initial_weights_filename)
        else:
            neural_network = tools.NeuralNetwork(nodes).to(tools.get_device())
        print("Nodes in layers: " + str(nodes))
        print("Learning rate: " + str(learning_rate[i]))
        print("Training cycles: " + str(training_cycles))

        # Save initial weights if save weights
        if save_initial_weights:
            neural_network.save_weights(path + initial_weights_filename)

        # Train neural network on training set
        if should_train:
            train_time = time.time()
            training_MSE_loss, training_percent_loss, n_unclean_points = neural_network.train_on_data(train_data, train_targets,epochs=training_cycles,lr=learning_rate[i])
            train_time = time.time() - train_time
            print("Unclean points: " + str(n_unclean_points))

        # Save errors as CSV file
        tools.tensor_to_csv(training_MSE_loss, (path + "MSE_loss_LR" + str(learning_rate[i])),fieldnames=['Data point', 'Error'])
        tools.tensor_to_csv(training_percent_loss, (path + "percent_loss_LR" + str(learning_rate[i])),fieldnames=['Data point', 'Error'])

        # Save final weights if save weights
        if save_final_weights:
            neural_network.save_weights(path + final_weights_filename)

        print("Estimated car price of first car (a.k.a. data point): " + str(neural_network(train_data[0]).item()))
        print("Supposed to be: " + str(train_targets[0].item()))

        # Test neural network on test set, log errors
        test_time = time.time()
        testing_loss = neural_network.test_on_data(test_data, test_targets,eval_method=test_eval_method)
        test_time = time.time() - test_time
        
        # If we save test results, save them
        if save_errors:
            tools.tensor_to_csv(testing_loss, path + error_file)

        # Plot results with learning rate in filenames
        print("Plotting results..............",end="")
        tools.plot_results(training_MSE_loss,filename=path+f"FigTrainingMSEloss{learning_rate[i]}", xlabel="Training cycles", ylabel="MSE error",title=("Training loss, learning rate: " + str(learning_rate[i])), show_plot=show_plots)
        tools.plot_results(100*training_percent_loss,filename=path+f"FigTrainingpercentloss{learning_rate[i]}", xlabel="Training cycles", ylabel="% error",title=("Training loss, learning rate: " + str(learning_rate[i])), show_plot=show_plots)
        tools.plot_results(100*testing_loss,filename=path+f"Figaftertrainingtestloss{learning_rate[i]}", ylabel="% error",title=("After training " + str(training_cycles) + " cycles, learning rate: " + str(learning_rate[i])), show_plot=show_plots)
        print(Fore.GREEN + "Complete" + Style.RESET_ALL)

        # Run final messages and print times
        print("\nTraining took {:0.1f} minutes with a train ratio of {:0.1f} %".format(train_time/60, 100*train_ratio))
        print("Testing took {:0.1f} seconds".format(test_time))
        print("Mean error: {:.1f} %".format(100*torch.mean(testing_loss).item()))
        print("Max error: {:.1f} %".format(100*torch.max(testing_loss).item()))
        
        layer_stacks_mean_errors.append(torch.mean(testing_loss).item())
        layer_stacks_max_errors.append(torch.max(testing_loss).item())
        
        print(Fore.YELLOW + "BIG RUN STATUS: {:.1f}".format(100*(layer_stack_ID*len(learning_rate)+i)/(num_layer_stacks*len(learning_rate))) + " %" + Style.RESET_ALL)

# Find minimum mean_error and minimum max_error, print which neural network reached it
print("Best mean_error: " + str(100*min(layer_stacks_mean_errors)))
index = layer_stacks_mean_errors.index(min(layer_stacks_mean_errors))
print("From layer stack: " + str(layer_stacks[index]) + " on run " + str(index+1))
print("Best max_error: " + str(100*min(layer_stacks_max_errors)))
index = layer_stacks_max_errors.index(min(layer_stacks_max_errors))
print("From layer stack: " + str(layer_stacks[index]) + " on run " + str(index+1))

# Calculate runtime and print successful run message
runtime = time.time() - start_time
print("\nmain.py ran successfully in " + Fore.BLUE + " {:0.1f} minutes".format(runtime/60)  + Style.RESET_ALL + " at " + datetime.datetime.now().strftime("%H:%M:%S") + "\n")
input("Press enter to end") # In case a run is being done in a terminal window then add input prompt so window doesn't close automatically