######IMPORTS
import torch
##################################
"""
This file contains the CNN used for this ML project
The Structure of the CNN is defined in the __init__ function
The forward pass through the CNN is processed by calling the forward() function
"""



#Network
class CNN(torch.nn.Module):
    def __init__(self, n_in_channels: int = 1, n_hidden_layers: int = 3, n_kernels: int = 32, kernel_size: int = 7):
        """Simple CNN with `n_hidden_layers`, `n_kernels`, and `kernel_size` as hyperparameters"""
        super().__init__()
        

        cnn = []

        for i in range(n_hidden_layers):
            cnn.append(torch.nn.Conv2d(
                in_channels=n_in_channels,
                out_channels=n_kernels,
                kernel_size=kernel_size,
                padding=int(kernel_size / 2)
            ))
            cnn.append(torch.nn.ReLU())
            n_in_channels = n_kernels


        self.hidden_layers = torch.nn.Sequential(*cnn)
        
        self.output_layer = torch.nn.Conv2d(
            in_channels=n_in_channels,
            out_channels=3,
            kernel_size=kernel_size,
            padding=int(kernel_size / 2)
        )

    
    def forward(self, x: torch.Tensor):
        """Apply CNN to input `x` of shape (N, n_channels, X, Y), where N=n_samples and X, Y are spatial dimensions"""
        #x = x.reshape([1,3,100,100])
        #print(len(self.hidden_layers))
        cnn_out = self.hidden_layers(x)  # apply hidden layers (N, n_in_channels, X, Y) -> (N, n_kernels, X, Y)
        #expected scalar type Byte but found Float
        pred = self.output_layer(cnn_out)  # apply output layer (N, n_kernels, X, Y) -> (N, 1, X, Y)
        #pred = cnn_out
        return pred



