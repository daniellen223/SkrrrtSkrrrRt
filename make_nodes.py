# Generate the structure for each layer with rows starting in descending order
# Set max number of rows
n_layers = 11

# For each number of hidden layers. From 1 layer to 11 layers in hidden layers
'''
for n_layers in range(n_rows):
    # Init empty current_layer
    layer_stack = []
'''

# For each version of the layer stack update it and print
for layers_in_stack in range(2):
    # Init empty layer_stack
    layer_stack = []
    # Make each node number for each layer
    for layer_num in range(1, n_layers+1):
        layer = []
        for each_layer_in_stack in range(layers_in_stack):
            # Find how many nodes should be in this layer
            n_nodes = 2**(layer_num)
            layer.append(n_nodes)  # Append rows as they are
        layer_stack.append(layer)

    # Print results
    print("Layers in stack: " + str(layers_in_stack))
    print(layer_stack)


