"""Small, independently callable examples of PyTorch tensor operations."""

import torch


def show_tensors(device):
    print(torch.__version__)
    print(torch.tensor([[1, 2], [3, 4]]))
    print(torch.tensor([[1, 2], [3, 4]], device=device))
    print(torch.tensor([[1, 2], [3, 4]], dtype=torch.float64))


def show_numpy_conversion(device):
    tensor = torch.tensor([[1, 2], [3, 4]])
    print(tensor.numpy())
    tensor = torch.tensor([[1, 2], [3, 4]], device=device)
    print(tensor.to("cpu").numpy())


def show_indexing():
    tensor = torch.FloatTensor([1, 2, 3, 4, 5, 6, 7])
    print(tensor[0], tensor[1], tensor[-1])
    print(tensor[2:5], tensor[4:])


def show_addition():
    first = torch.tensor([1, 2, 3])
    second = torch.tensor([4, 5, 6])
    print(first + second)


def show_shape_operations():
    tensor = torch.arange(10)
    print(tensor)
    print(tensor.view(2, 5))
    first = torch.Tensor([1, 2, 3])
    second = torch.Tensor([4, 5, 6])
    print(torch.stack((first, second)))
    print(torch.cat((first, second)))
    print(torch.Tensor([[1, 2, 3], [4, 5, 6]]).t())


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    show_tensors(device)
    show_numpy_conversion(device)
    show_indexing()
    show_addition()
    show_shape_operations()

if __name__ == "__main__":
    main()
