# -*- coding: utf-8 -*-
# Source notebook: Reinforcement Learning/Week1/2-6_Normalization.ipynb
# Code cells are preserved in notebook order; markdown and outputs are omitted.
# Notebook shell commands are retained as comments for use from a terminal.

# %% [code cell 2]
from sklearn.preprocessing import MinMaxScaler, StandardScaler
X1 = [[1, 2],
     [2, 1],
     [3, 2],
     [4, 3]]
X2 = [[1.5, 2864, 2.3],
      [2.6, 8372, 1.8],
      [1.2, 6453, 2.2],
      [2.3, 9587, 3.7],
      [1.9, 2332, 3.1],
      [3.7, 8574, 1.5],
      [2.1, 7665, 2.3],
      [1.4, 2428, 1.8],
      [3.7, 9476, 3.2],
      [1.5, 3422, 2.4]]

# %% [code cell 3]
scaler1 = MinMaxScaler()
scaler1.fit(X1)
nX1 = scaler1.transform(X1)
print(nX1)

# %% [code cell 4]
scaler2 = MinMaxScaler()
scaler2.fit(X2)
nX2 = scaler2. transform(X2)
print(nX2)
