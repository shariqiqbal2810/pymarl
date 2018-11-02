import torch


class FFAgent(torch.nn.Module):
    def __init__(self, input_shape, args):
        torch.nn.Module.__init__(self)
        self.args = args
        self.init_nonzeros = getattr(args, "input_initialization_nonzeros", 0)

        # Construct a list with all relevant input-output dimensions
        layer_dims = getattr(args, "ff_layer_dims", [])
        layer_dims.insert(0, input_shape)
        layer_dims.append(args.n_actions)

        # Constructs the fully connected layers from layer_dims
        self.layers = []
        for d in range(len(layer_dims) - 1):
            self.layers.append(torch.nn.Linear(layer_dims[d], layer_dims[d+1]))
            # Register this layer's parameters
            self.register_parameter("weight%u" % d, self.layers[-1].weight)
            self.register_parameter("bias%u" % d, self.layers[-1].bias)

        # Initialise first layer with a fixed variance, if specified
        if self.init_nonzeros > 0:
            std = self.init_nonzeros ** -0.5
            self.layers[0].weight.data.normal_(std=std)
            self.layers[0].bias.data.normal_(std=std)

    def init_hidden(self):
        """ Initialise a dummy. """
        return self.layers[0].weight.new(1).zero_()

    def forward(self, inputs, hidden_state):
        """ Computes the output with a number of hidden layers specified by <ff_layer_dims>. """
        x = inputs
        for i in range(len(self.layers) - 1):
            x = torch.nn.functional.relu(self.layers[i](x))
        # No ReLu for the last layer
        x = self.layers[-1](x)
        return x, hidden_state
