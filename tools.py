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
import matplotlib.pyplot # For plotting results
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

# Plot results
def plot_results(results: torch.Tensor, filename: str = None):
    '''
    Plots the fig and saves it if a filename is given
    
    input:
    filename    : The filename to save as without a filetype ending
    '''
    # fig = matplotlib.pyplot.figure()
    # Find number of data points and make x_values
    N = results.size()[0]
    x_values = range(N)

    # Detach the tensor if it requires gradients
    if results.requires_grad:
        results = results.detach()
        
    # Convert the tensor to a NumPy array
    y_values = results.numpy()

    matplotlib.pyplot.plot(x_values,y_values)
    matplotlib.pyplot.title("Error", fontsize='16')	#title
    matplotlib.pyplot.xlabel("data points",fontsize='13')	#adds a label in the x axis
    matplotlib.pyplot.ylabel("Error",fontsize='13')	#adds a label in the y axis
    matplotlib.pyplot.grid()	#shows a grid under the plot
    if filename != None:
        matplotlib.pyplot.savefig(filename + ".png")	#saves the figure in the present directory
    matplotlib.pyplot.show()

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
        # WHAT IS THIS? Was copied from tutorial
        super().__init__()
        # WHAT IS THIS?  Was copied from tutorial
        self.flatten = torch.nn.Flatten()
        # Create layer_stack containing instructions for each layer
        self.linear_layer_stack = torch.nn.Sequential(
            torch.nn.Linear(D, M),  # First linear combination between each dimension and weights to hidden layer
            torch.nn.ReLU(),        # ReLU activation function inside hidden layer
            torch.nn.Linear(M, 1),  # Last output is 1 value which should be the car price
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

    # Train neural network
    # Based on https://pytorch.org/tutorials/beginner/introyt/trainingyt.html ????? At least if it helps
    def train_on_data(self, train_data: torch.Tensor, train_targets: torch.Tensor, epochs: int=100, lr: float=0.0001) -> torch.Tensor:
        '''
        Trains the neural network with the given training data and targets by:
        1. forward propagating an input feature through the network
        2. Calculate the error between the prediction the network made and the actual target
        3. Backpropagating the error through the network to adjust the weights.
        
        inputs:
        train_data      : Size (N x D) where N is the number of data points and D is the number of dimensions. The training data.
        train_targets   : Size (N x 1) where N is the number of data points. The targets, price for each car.
        epochs          : Number of epochs that the training will run
        lr              : Learning rate
        
        output:
        loss_matrix     : Size (epochs) where the first value is the running loss after propagating through the whole train_data, the second value is the running loss after the second epoch etc.
        
        Possible more inputs:
        - Momentum
        - More?
        '''
        # N number of data points and D number of dimensions
        N, D = train_data.size()
        #print("\nTrain data size: " + str(train_data.size()))
        #print("\nTrain targets size: " + str(train_targets.size()))

        # Modified code from ChatGPT start
        # -------------------------------------------------------------------
        # Define the loss function (Mean Squared Error for regression)
        MSE_Loss = torch.nn.MSELoss()

        # Define the optimizer (Adam optimizer)
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        
        # Set model to training mode
        self.train()
        
        # Batch training data into loader
        # train_loader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True)
        
        # Initialize loss_matrix
        loss_matrix = torch.zeros(epochs)
        
        # Training loop through each epoch
        for epoch in range(epochs):
            # Reset runnin_loss
            running_loss = 0.0
            # Print status
            print(100*epoch/epochs,"%\r",end="Training neural network.......")
            # Loop through each data point
            for n in range(N):
                # Zero the gradients
                optimizer.zero_grad()

                # Forward pass
                car_price = self(train_data[n,:])
                # Calculate loss
                loss = MSE_Loss(train_targets[n].unsqueeze(0), car_price) # Unsqueeze so that the tensor sizes match                
                # Backward pass and optimization
                loss.backward()
                optimizer.step()
                # Add loss to running loss
                running_loss += loss.item()
            # End for n
            # Log running_loss to loss_matrix
            loss_matrix[epoch] = running_loss
        # End for epoch
            
        # Modified code from ChatGPT end
        # -------------------------------------------------------------------
        
        # Set neural network back to evaluation mode
        self.eval()
        
        # Return loss_matrix
        return loss_matrix

    # Test neural network
    # Based on self.train_on_data()
    def test_on_data(self, test_data: torch.Tensor, test_targets: torch.Tensor) -> torch.Tensor:
        '''
        Tests the neural network with the given testing data and targets by:
        1. forward propagating test_data through the network
        2. Calculate the error between the prediction the network made and the actual target
        
        inputs:
        test_data      : Size (N x D) where N is the number of data points and D is the number of dimensions. The training data.
        test_targets   : Size (N x 1) where N is the number of data points. The targets, price for each car.
        
        output:
        loss_matrix     : Size (N) where N is the number of data points and each value is the error between the test_target and the neural networks guess using test_data.
        '''
        # N number of data points and D number of dimensions
        N, D = test_data.size()

        # Define the loss function (Mean Squared Error for regression)
        MSE_Loss = torch.nn.MSELoss()
        
        # Set model to evaluation mode just in case
        self.eval()
                
        # Initialize loss_matrix
        loss_matrix = torch.zeros(N)
        
        # Status print message index
        n_print = int(N/100)
        
        # Loop through each data point and find error
        for n in range(N):
            # Forward pass to estimate value
            car_price = self(test_data[n,:])
            # Calculate and log error
            loss_matrix[n] = MSE_Loss(test_targets[n].unsqueeze(0), car_price) # Unsqueeze so that the tensor sizes match                

            # Print status for every whole percent
            if  n % n_print == 0:
                print("{:.1f} %\r".format(100*n/N),end="Testing neural network........")

        # End for n
        # Return loss_matrix
        return loss_matrix
