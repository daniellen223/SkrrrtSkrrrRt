# Generate the structure for each layer with rows starting in descending order
# Set max number of layer_stacks
n_layers = 2

print("Starting")

layer_stacks = []

# For each version of the layer stack update it and print
for n_layers_in_stack in range(1, n_layers+1):
    # Print results
    print("Layers in stack: " + str(n_layers_in_stack))
    # Make each node number for each layer
    for layer_num in range(1, n_layers - n_layers_in_stack + 2):
        # Init empty layer_stack
        layer_stack = []
        for each_layer_in_stack in range(n_layers_in_stack):
            # Find how many nodes should be in this layer
            n_nodes = 2**(layer_num + n_layers_in_stack - each_layer_in_stack - 1)
            # Append layers to layer_stack
            layer_stack.append(n_nodes)

        layer_stacks.append(layer_stack)
        print(layer_stack)

print("Done")