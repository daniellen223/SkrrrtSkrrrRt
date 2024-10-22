'''
Authors:    Ragnar, Danielle and Huldar
Date:       2024-10-08
Project:
Background functions for helping solve the kaggle used car prices competition:
https://www.kaggle.com/competitions/playground-series-s4e9/data
'''
# Imports
import torch # To work with tensors and neural network functions
from typing import Union # To return multiple values as a union
import pandas as pd # For loading data
from sklearn.preprocessing import LabelEncoder # For loading data

# Read data function
def read_in_file(filename: str = 'train.csv') -> Union[torch.Tensor, torch.Tensor] :
    '''
    Reads in the data from a csv file, given the filename.
    Formats the data in a workable format and splits it into data and targets.
    Returns the data and targets.
    Copied from readinfiles.py the 2024-10-20
    
    input:
    filename    : The filename to fetch
    
    outputs:
    data_tensor    : Size (N x 12) where N is the number of data points
    target_tensor : Size (N) where N is the number of data points. Cont
    
    data_tensor column meanings:
    0 : 
    1 : 
    2 :
    3 : 
    4 : 
    5 : 
    6 : 
    7 : 
    8 : 
    9 : 
    10 : 
    11 : 
    '''
    # Try except in case something goes wrong
    try:
        # Load the data
        data = pd.read_csv(filename)

        # Define categorical columns for label encoding
        categorical_columns = ['brand', 'model', 'fuel_type', 'transmission', 'engine', 'ext_col', 'int_col']

        # Initialize the label encoder
        label_encoder = LabelEncoder()

        # Apply label encoding to each categorical column
        for col in categorical_columns:
            data[col] = label_encoder.fit_transform(data[col])

        # Convert binary columns manually (e.g., 'Yes'/'No' or similar binary data)
        binary_columns = ['accident', 'clean_title']
        data['accident'] = data['accident'].map({'None reported': 0, 'At least 1 accident or damage reported': 1})
        data['clean_title'] = data['clean_title'].map({'Yes': 1, 'No': 0})

        # Convert any remaining object columns to numeric values
        object_columns = data.select_dtypes(include=['object']).columns
        if len(object_columns) > 0:
            print("Non-numeric columns found, attempting to convert:", object_columns)
            data[object_columns] = data[object_columns].apply(lambda x: pd.to_numeric(x, errors='coerce'))

        # Ensure 'price' is numeric
        data['price'] = pd.to_numeric(data['price'], errors='coerce')

        # Separate features and target
        arraydata = data.iloc[:, :-1]  # Assuming the last column is the target
        arraytarget = data['price']  # Explicitly use the 'price' column

        # Convert to tensor
        data_tensor = torch.tensor(arraydata.values, dtype=torch.float32)
        target_tensor = torch.tensor(arraytarget.values, dtype=torch.float32)

        # Return data_tensor and target_tensor
        return data_tensor, target_tensor
    
    # Except statement
    except Exception as e:
        raise("Error reading data:\n" + str(e))

# Split data function
def split_data(
    data: torch.Tensor,
    targets: torch.Tensor,
    train_ratio: float = 0.8,
    shuffle: bool = False) -> Union[tuple, tuple]:
    '''
    Splits the data and targets into a train and test set with the train_ratio.
    
    Copied from assignment 02_classification tools.py and modified for pytorch instead of numpy
    
    inputs:
    data        : Size (N X D). The data points, tensor where N is the number of data points and D is how many dimensions each data point has
    targets     : Size (N x 1). The targets, tensor where N is the number of data points, includes the real value the neural network tries to estimate
    train_ratio : Value from 0 to 1 (both included) accepted. How high ratio of the inputted data should be used for training, the rest is used for testing
    shuffle     : If the data should be shuffled before splitting.

    outputs:
    train_data      : Size (split_index x D). The training data. split_index is N*train_ratio and D is how many dimensions each data point has,
    train_targets   : Size (split_index x 1). The training targets.
    test_data       : Size ((N-split_index) x D). The testing data
    test_targets    : Size ((N-split_index) x 1). The testing targets
    '''
    # Try except in case something goes wrong
    try:
        # validate train_ratio, if out of bounds, throw error
        if train_ratio < 0 or train_ratio > 1:
            raise ValueError("train_ratio must be a value from 0 to 1 (both included)")
        
        # Find N
        N = data.shape[0]
        
        # If shuffle is True, shuffle data and targets with the same random shuffle
        if shuffle:
            # Get new indices with random permutation
            indices_new = torch.randperm(N)
            # Shuffle data and targets with new indices
            data = data[indices_new]
            targets = targets[indices_new]
            
        # Find split_index between training data and test data.
        split_index = int(N * train_ratio)

        # Split data into training and testing set. 2 different ways, one way for 1 dimensional data and one way for 2 dimensional data
        if len(data.shape) > 1:
            train_data  =   data[0:split_index, :]
            train_targets   =   targets[0:split_index]
            test_data   =   data[split_index:, :]
            test_targets    =   targets[split_index:]
        else:
            train_data  =   data[0:split_index]
            train_targets   =   targets[0:split_index]
            test_data   =   data[split_index:]
            test_targets    =   targets[split_index:]

        # Returned split training and testing sets
        return (train_data, train_targets), (test_data, test_targets)

    # Except statement
    except Exception as e:
        raise("Data splitting error: " + str(e))

#Making weights
def init_weights(D: int, M: int, bias: bool = True) -> Union[torch.Tensor, torch.Tensor]:
    ''' 
    Inputs
    D : Number of dimensions
    M : Number of nodes in the hidden layer
    '''
    # making w1
    if bias:
        w1 = torch.rand(D+1)
    else:
        w1 = torch.rand(D)

    if bias: 
        w2 = torch.rand(M+1)
    else:
        w2 = torch.rand(M)

    return w1, w2

# Measure error function


# Activation function, h
def h(z1: torch.Tensor, w1: torch.Tensor) -> float:
    '''
    ReLU neuron activation function.
    The same function is used for each neuron.
    Does a linear combination of the previous layers outputs and the weights between those nodes and this one
    
    inputs:
    z1  :   Size (1 x D), the output of the last layer.
    w1  :   Size (D x 1), the weights for this neuron.
    
    outputs:
    z2  :   the ReLU of the dot product of z1 and w1, the output of each node of this layer
    '''
    # Get weighed sum (dot product) of z1 and w1
    w_sum = torch.matmul(z1,w1)
    # ReLU of the weighed sum
    ReLU = torch.nn.ReLU()
    return ReLU(w_sum).item() # .item() pulls the 1x1 tensor value out


# Get device for torch
# Based on https://pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html
def get_device() -> str:
    '''
    Gets which device the neural network will run on
    
    output:
    device: The device to run the neural network model on
    '''
    # Default to cpu"
    device = "cpu"
    # If cuda available, use cuda
    if torch.cuda.is_available():
        device = "cuda"
    # If mps available, use mps
    elif torch.backends.mps.is_available():
        device = "mps"

    # Return device
    return device

# Make single layer neural network class
# Based on https://pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html
class NeuralNetwork(torch.nn.Module):
    '''
    Class for neural network, contains layers and weights.
    '''
    # Initialize network
    def __init__(self, D, M):
        '''
        Initializes the neural network with random initial weights
        
        inputs:
        D   : Number of dimensions in the dataset
        M   : Number of hidden layer nodes
        '''
        super().__init__()
        self.flatten = torch.nn.Flatten()
        # Create layer_stack containing instructions for each layer
        self.linear_layer_stack = torch.nn.Sequential(
            torch.nn.Linear(D, M),  # First linear combination between each dimension and weights to hidden layer
            torch.nn.ReLU(),        # ReLU activation function inside hidden layer
            torch.nn.Linear(M, 1),  # Last output is 1 value which should be the car price
            torch.nn.ReLU(),        # ReLU activation function on output
        )

    def forward(self, car):
        '''
        Forward propagation through each layer of the neural network
        
        input:
        car : Input datapoint with all the car information
        
        output:
        car_price   : Estimated price of inputted car
        '''
        # car = self.flatten(car) # Used for multilayer input data, like images with RGB values (3 layers)
        # Run the input data through the layers, return output
        car_price = self.linear_layer_stack(car)
        return car_price

    def train(self, train_data: torch.Tensor, train_targets: torch.Tensor):
        '''
        Trains the neural network using the given training data and targets
        
        inputs:
        train_data      : Size (N x D) where N is the number of data points and D is the number of dimensions. The training data.
        train_targets   : Size (N x 1) where N is the number of data points. The targets, price for each car.
        '''
        
        raise ReferenceError("Error: neural network train function has not yet been implemented. Huldar 2024-10-22")

