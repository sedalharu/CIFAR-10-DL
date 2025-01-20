# CIFAR-10-DL

This project involves implementing a Convolutional Neural Network (CNN) using PyTorch to classify images from the CIFAR-10 dataset. It was undertaken as part of a study on deep learning.

## Structure

```
├── data/              # Dataset directory
├── data.py            # Data loading and preprocessing
├── model.py           # CNN model architecture
├── train.py           # Training implementation
├── evaluate.py        # Evaluation and metrics
└── main.py            # Main execution script
```

## Features

- Custom CNN architecture with 3 convolutional layers
- Data augmentation (Random Crop, Horizontal Flip)
- Batch Normalization and Dropout for preventing overfitting
- Performance visualization (training curves, confusion matrix)
- Model evaluation with various metrics

## Requirements

- Python = 3.8.10
- PyTorch = 2.0.0
- torchvision = 0.15.0
- scikit-learn = 1.2.0
- matplotlib = 3.7.0
- seaborn = 0.12.0

## Installation

GPU support (CUDA 11.8):
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

## Usage

Run the training and evaluation:
```bash
python main.py
```

## Model Architecture

- Input: 3x32x32 (CIFAR-10 images)
- 3 Convolutional layers with BatchNorm and ReLU
- MaxPooling after each conv layer
- 2 Fully connected layers with Dropout
- Output: 10 classes

## Training Details

- Epochs: 150
- Learning Rate: 0.001
- Optimizer: Adam
- Loss Function: Cross Entropy
- Batch Size: 64

## Results

The model generates:
- Training loss and accuracy curves
- Confusion matrix
- Class-wise precision, recall, and F1-scores
- Overall model performance metrics

## Output

- `best_model.pth`: Best performing model weights
- `training_results.png`: Training curves
- `confusion_matrix.png`: Confusion matrix visualization
- `class_performance.png`: Class-wise performance metrics
- `evaluation_results.txt`: Detailed evaluation metrics
