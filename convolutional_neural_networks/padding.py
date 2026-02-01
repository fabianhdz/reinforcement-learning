"""Demonstrate how convolution padding affects output shapes."""

import torch
import torch.nn as nn


def apply_convolution(inputs, *, padding=1):
    convolution = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=padding)
    return convolution(inputs)


def main():
    inputs = torch.randn(1, 3, 32, 32)
    outputs = apply_convolution(inputs)
    print("Input tensor shape:", inputs.shape)
    print("Output tensor shape:", outputs.shape)

if __name__ == "__main__":
    main()
