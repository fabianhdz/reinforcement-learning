# -*- coding: utf-8 -*-

# %%
import torch
import torch.nn as nn

# Create a 2D convolutional layer with padding
conv_layer = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)

# Create a random input tensor
input_tensor = torch.randn(1, 3, 32, 32)  # Batch size of 1, 3 channels, 32x32 image

# Apply the convolutional layer to the input tensor
output_tensor = conv_layer(input_tensor)

# Print the shapes of the input and output tensors
print("Input tensor shape:", input_tensor.shape)
print("Output tensor shape:", output_tensor.shape)
