# DSN-pytorch-talk

A hands-on introduction to neural networks in PyTorch, presented as a live coding talk. The accompanying notebook (`DSN_NN_Talk.ipynb`) builds, trains, and visualizes a small feedforward network that learns to separate non-linearly separable circle data.

The goal is to demystify the core PyTorch workflow: prepare data, define a model, define loss and optimizer, write a training loop, evaluate, and visualize what the model learned.

Note: Presentation slides for the talk can be found @ https://canva.link/bayf1yzow7osuux

## What this covers

1. Device-agnostic setup (CUDA if available, otherwise CPU)
2. Synthetic data generation with `sklearn.datasets.make_circles`
3. Conversion to PyTorch tensors and train/test split
4. A 3-layer `nn.Module` binary classifier with ReLU activations
5. Training with `BCEWithLogitsLoss` and SGD
6. Tracking train/test loss and accuracy over epochs
7. Plotting loss/accuracy curves and the model decision boundary
8. A brief discussion of overfitting when training for many epochs

## Notebook walkthrough

### 1. Imports and device setup

Sets up `torch`, `nn`, `matplotlib`, `sklearn`, and `numpy`, then selects:

```python
device = "cuda" if torch.cuda.is_available() else "cpu"
```

All tensors and the model are later moved to this device so the same code runs on GPU and CPU.

### 2. Data generation and visualization

Generates 1000 samples with:

```python
X, y = make_circles(n_samples=1000, noise=0.03, random_state=42)
```

This creates two concentric circles, a classic example of data that a linear model cannot separate. The data is visualized with a `matplotlib` scatter plot colored by class label.

### 3. Data pre-processing

Converts NumPy arrays to `float32` PyTorch tensors, splits 80/20 into train and test with `train_test_split`, and moves all splits to the target device.

Resulting shapes:

- `X_train: [800, 2]`
- `y_train: [800]`
- `X_test: [200, 2]`
- `y_test: [200]`

### 4. Reusable metric plotter

Defines `plot_metrics(history_dict)` to plot train/test loss and train/test accuracy side by side, plus a small `accuracy_fn` helper that computes:

```python
correct / len(y_pred) * 100
```

### 5. Building the model

Defines `CircleModel(nn.Module)`:

- `Linear(2, 10)` + ReLU
- `Linear(10, 10)` + ReLU
- `Linear(10, 1)` (raw logits output, no sigmoid)

The non-linear ReLU activations are essential. Without them the stacked linear layers would collapse to a single linear function and could not learn the circular boundary.

Loss and optimizer:

```python
loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.SGD(model_0.parameters(), lr=0.1)
```

`BCEWithLogitsLoss` combines sigmoid and binary cross-entropy in one numerically stable step, which is the standard choice for binary classification with logits output.

### 6. Training loop

Trains for 1500 epochs with the standard PyTorch pattern per epoch:

Train:
1. Forward pass: `y_logits = model_0(X_train).squeeze()`
2. Convert logits to labels: `torch.round(torch.sigmoid(y_logits))`
3. Compute loss and accuracy
4. `optimizer.zero_grad()`
5. `loss.backward()`
6. `optimizer.step()`

Evaluate:
1. `model_0.eval()` + `torch.inference_mode()`
2. Forward pass on test data
3. Compute test loss and accuracy

Metrics are logged every 100 epochs. Example output from the notebook:

- Epoch 0: Loss 0.705, Acc 50.00% | Test Loss 0.707, Test Acc 50.00%
- Epoch 700: Loss 0.609, Acc 78.38% | Test Loss 0.630, Test Acc 71.00%
- Epoch 1000: Loss 0.266, Acc 99.75% | Test Loss 0.308, Test Acc 99.50%
- Epoch 1400: Loss 0.045, Acc 100.00% | Test Loss 0.074, Test Acc 100.00%

The notebook notes that pushing to 1500 epochs gives the model more training time but raises suspicion of overfitting, since train and test performance both saturate near 100 percent on this simple synthetic set.

### 7. Visualizing metrics

Calls `plot_metrics(history)` to show loss decreasing and accuracy rising over epochs for both train and test splits.

### 8. Visualizing the decision boundary

Defines `plot_decision_boundary(model, X, y)` which:

1. Moves the model and data to CPU for `matplotlib`
2. Builds a 101x101 meshgrid over the feature space
3. Predicts a class for every grid point with `torch.inference_mode()`
4. Draws the predicted regions with `contourf` and overlays the true test points

This is the payoff visual: you can see the model has learned a roughly circular boundary separating the inner and outer rings.

## How to run

Requirements:

- Python 3.9+
- torch
- scikit-learn
- matplotlib
- numpy
- jupyter

Install:

```bash
pip install torch scikit-learn matplotlib numpy jupyter
```

Run:

```bash
jupyter notebook DSN_NN_Talk.ipynb
```

Run cells top to bottom. If you have an NVIDIA GPU with CUDA, training will use it automatically. Otherwise it falls back to CPU.

## Key takeaways

- Non-linear activations are what let neural networks solve non-linear problems.
- `BCEWithLogitsLoss` is the standard loss for binary classification from logits.
- The train/eval + `zero_grad` / `backward` / `step` pattern is the core of every PyTorch training loop.
- Always evaluate in `eval()` mode under `inference_mode()` so dropout/batchnorm behave correctly and gradients are disabled.
- Plotting loss, accuracy, and the decision boundary makes model behavior concrete.
- More epochs is not always better. Watch the gap between train and test metrics for signs of overfitting.

## Source materials

Each source below is linked to the notebook section it supports. All picks are beginner-friendly explainers rather than dense API reference.

- freeCodeCamp, PyTorch for Deep Learning Full Course (Daniel Bourke) - gentle full walkthrough of tensors, device setup (`cuda` vs `cpu`), `nn.Module`, loss, optimizer, and the train/test loop. Used for Sections 1, 3, 5, and 6.
  https://www.youtube.com/watch?v=V_xro1VeRhA
- Learn PyTorch for Deep Learning, Chapter 01: PyTorch Workflow - `accuracy_fn`, `plot_decision_boundary`, the `CircleModel` style architecture, and the full circles training workflow this talk is structured around. Used for Sections 4 through 8.
  https://www.learnpytorch.io/01_pytorch_workflow/
- Learn PyTorch for Deep Learning, Chapter 02: Neural Network Classification - binary classification on circles/moons data, `BCEWithLogitsLoss` intuition, and decision boundary reading. Used for Sections 2, 5, and 8.
  https://www.learnpytorch.io/02_pytorch_classification/
- 3Blue1Brown, Neural Networks series - visual intuition for layers, non-linearities, and why a straight line cannot separate circle data. Used for Sections 2 and 5.
  https://www.3blue1brown.com/topics/neural-networks
- Google Machine Learning Crash Course: Training and Test Sets - why you hold out a test split and how to read train vs test metrics. Used for Sections 3, 6, and 7.
  https://developers.google.com/machine-learning/crash-course/training-and-test-sets/splitting-data
- Google Machine Learning Crash Course: Overfitting - gentle visual explainer for why more epochs can hurt generalization. Used for the Section 6 overfitting note.
  https://developers.google.com/machine-learning/crash-course/overfitting/overfitting
