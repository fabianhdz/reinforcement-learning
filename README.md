# Reinforcement Learning

Code from my reinforcement learning coursework, organized by topic. The collection also includes the supervised learning, deep learning, and transformer exercises covered alongside RL.

This first import preserves the original implementations, including unfinished exercises, repeated examples, environment APIs, and existing path assumptions. Existing Python files and Python scripts extracted from the DQN ZIP are copied byte for byte. Notebook code is exported to `.py` in its original cell order with `# %%` markers. The four notebook `!pip install` lines are kept as comments so the exports use valid Python syntax; their commands can be run manually in a terminal without the leading `!`.

## Topics

| Section | Contents | Python files |
| --- | --- | --- |
| [Supervised learning](supervised_learning/) | Regression, gradient descent, normalization, linear classification, logistic regression, and multiclass classification. | 8 |
| [PyTorch fundamentals](pytorch_fundamentals/) | Tensors, data loading, model definitions, parameter settings, and training examples. | 4 |
| [Neural networks](neural_networks/) | XOR, fully connected MNIST models, activation functions, classification examples, and dropout. | 4 |
| [Convolutional neural networks](convolutional_neural_networks/) | FashionMNIST, a CNN MNIST exercise, and convolution padding. | 3 |
| [Monte Carlo learning](monte_carlo/) | GridWorld prediction and control using Monte Carlo returns. | 2 |
| [Temporal-difference learning](temporal_difference/) | GridWorld TD prediction, SARSA, and Q-learning. | 3 |
| [Function approximation](function_approximation/) | Linear and deep neural network function approximation examples. | 2 |
| [Deep Q-learning](deep_q_learning/) | The notebook DQN example and three CartPole scripts recovered from the source ZIP. | 4 |
| [Policy gradients](policy_gradients/) | REINFORCE and policy-based CartPole examples, including the modified notebook version. | 4 |
| [Actor-critic learning](actor_critic/) | Original and Gymnasium actor-critic notebook examples, plus the GPU and TensorBoard implementation. | 3 |
| [Proximal policy optimization](proximal_policy_optimization/) | The PPO implementation from Week 8. | 1 |
| [Transformers and sequence learning](transformers/) | Encoder-decoder transformer translation examples and the separate homework versions. | 12 |
| [Decision transformers and offline RL](decision_transformers/) | CartPole trajectory collection using PPO and decision transformer training and evaluation. | 2 |
| [Image segmentation](image_segmentation/) | U-Net training and evaluation on Oxford-IIIT Pet images and trimaps. | 1 |

The import contains **53 Python files**: 32 notebook exports, 18 existing scripts, and 3 scripts extracted from `Week4/dqn_cartpole.zip`. Each topic has its own Git commit after the initial repository setup commit. Alternate homework and experiment versions remain separate.

## Working with the code

These are independent coursework examples. Use an environment with the dependencies required by the example you choose, then run it from its own directory. Some exports contain multiple experiments that were originally run as separate notebook cells; the `# %%` markers allow selecting those cells in a supporting editor.

Dependencies used across the collection include NumPy, pandas, Matplotlib, scikit-learn, PyTorch, torchvision, Pillow, Gym, Gymnasium, TensorBoard, h5py, Stable-Baselines3, requests, and kagglehub. There is no pinned shared environment yet. Both Gym and Gymnasium APIs are preserved as originally written, and some examples require CUDA or other original runtime assumptions.

See each section's README for its files and data assumptions. In particular, transformer homework can require CSV data, decision transformer training requires collected trajectories, and image segmentation requires the Oxford-IIIT Pet dataset. Data files and model checkpoints remain in the original workspace or must be supplied separately. Training and evaluation have not been run as part of this organizational import.

## Source tracking

[SOURCE_MAP.csv](SOURCE_MAP.csv) maps every Python file to its original workspace path. It records the conversion type, exported code cell count, commented shell commands, and SHA-256 hashes of both the original source and repository file. For ZIP entries, the source hash describes the extracted Python file and `::` separates the archive path from its member name.

The original `Reinforcement Learning/` folder is outside this repository and is left unchanged. Lecture slides, documents, PDFs, notebook markdown and outputs, datasets, trained weights, plots, videos, logs, caches, and workspace settings are excluded. The research-paper and lab folders contain no Python scripts or code notebooks, so they do not have empty sections here.

Two notebooks are intentionally omitted:

- `Week2/5-7-lecture-note.ipynb`: Lecture-only notebook with no code cells.
- `Week1/.ipynb_checkpoints/slide3-chapter1.4-1.5-checkpoint.ipynb`: Exact duplicate code of Week1/1-3-code.ipynb.

## Import validation

All 53 Python files pass a syntax check without executing them. Source coverage, unchanged script bytes, exported notebook cell contents and order, and output hashes were checked during import. This verifies the organization and conversion; it does not establish that every original implementation runs correctly.
