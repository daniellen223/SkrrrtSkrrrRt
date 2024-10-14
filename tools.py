'''
Authors:    Ragnar, Danielle and Huldar
Date:       2024-10-08
Project:
Background functions for helping solve the kaggle used car prices competition:
https://www.kaggle.com/competitions/playground-series-s4e9/data
'''
# Imports
import torch

# Read data function


# Split data function



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

