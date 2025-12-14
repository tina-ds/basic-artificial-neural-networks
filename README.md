# Basic Artificial Neural Networks (MNIST)

This repository contains a practical assignment for the *Machine Learning* course.
The goal of the project is to implement a simple neural network framework from scratch
(using NumPy) and to compare different architectural and optimization choices on the MNIST dataset.

The work is inspired by the *Practical Deep Learning* course by Yandex School of Data Analysis.

---

## Project structure

- **main_notebook.ipynb**  
  The main notebook with experiments:
  - comparison of activation functions (ReLU, ELU, LeakyReLU, SoftPlus),
  - effect of Batch Normalization,
  - comparison of optimizers (momentum SGD vs Adam),
  - analysis of training stability and validation loss.

- **modules.ipynb**  
  Implementation of core neural network components from scratch:
  - layers (`Linear`, activations, `BatchNormalization`, `ChannelwiseScaling`),
  - loss function (`ClassNLLCriterion`),
  - optimization methods (momentum SGD, Adam),
  - container modules (`Sequential`).

- **mnist.py**  
  Utility module for loading the MNIST dataset from raw IDX files.

- **train-images-idx3-ubyte.gz**, **train-labels-idx1-ubyte.gz**  
  MNIST training data.

- **t10k-images-idx3-ubyte.gz**, **t10k-labels-idx1-ubyte.gz**  
  MNIST test data.

---

## Experiments and results

The experiments demonstrate that:

- Adam optimizer converges faster and more reliably than momentum SGD.
- Batch Normalization significantly stabilizes training and improves convergence,
  especially when using momentum SGD.
- ReLU provides a good balance between performance, stability, and computational efficiency,
  while activations involving exponential functions are more sensitive to scaling and
  numerical stability.

A PyTorch baseline model with Batch Normalization, Dropout, data augmentation, and Adam
reaches **~98.4% test accuracy** on MNIST.

---

## Requirements

- Python 3.x
- NumPy
- Matplotlib
- (Optional) PyTorch — used only for the baseline comparison

