import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from encoder import TransformerEncoder
from decoder import TransformerDecoder
from collections import Counter
import pandas as pd
import matplotlib.pyplot as plt


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(device)

file = "test_translation_data_100.csv"
df2 = pd.read_csv(file)

test_data = list(zip(df2['English'], df2['Spanish']))

