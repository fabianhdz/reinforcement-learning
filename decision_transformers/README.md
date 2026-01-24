# Decision transformers and offline RL

CartPole trajectory collection using PPO and decision transformer training and evaluation.

| Python file | Original source |
| --- | --- |
| [data_collection.py](data_collection.py) | `week11/data_collection.py` |
| [decision_transformer_encoder.py](decision_transformer_encoder.py) | `week11/decision_transformer_encoder.py` |

Original implementations and unfinished work are preserved. Notebook exports retain code cell order and code comments; lecture markdown and outputs are omitted.

`data_collection.py` produces `cartpole_data_50000.hdf5`, which the training script expects in its working directory. Existing datasets and checkpoints are excluded. Evaluation requires a model checkpoint.
