
# PyTorch Image Classification Project

This repository provides the necessary code and functions for setting up a PyTorch-based image classification pipeline. It includes utilities for data preparation, model building, training, testing, saving, and prediction, along with functions for handling various image transformations and visualizations.

## Table of Contents

1. [Overview](#overview)
2. [Modules](#modules)
3. [Installation](#installation)
4. [Usage](#usage)
5. [Model Architecture](#model-architecture)
6. [License](#license)

## Overview

This project is designed to help with the setup and management of an image classification task in PyTorch. It includes scripts for:

- Preparing datasets (including loading from Excel to directory structure)
- Creating data loaders with transformations
- Defining and training models
- Evaluating models
- Saving models
- Visualizing predictions and random images

## Modules

### 1. `data_setup.py`

Contains functions for setting up and loading image data:
- `create_dataloaders`: Creates PyTorch DataLoaders for training, testing, and validation datasets with the option to apply transformations.
- `image_from_Excel_to_folder`: Loads image data from a specified Excel file and organizes them into directories for training, testing, and validation.
- `display_random_images`: Plots random images from the dataset for visualization.

### 2. `utils.py`

Contains utility functions for saving and managing models:
- `save_model`: Saves a PyTorch model to a specified directory.

### 3. `engine.py`

Functions for training and evaluating a model:
- `train_step`: Executes a single training step for an epoch.
- `test_step`: Executes a single test step for an epoch.
- `train`: Manages the entire training and evaluation loop, storing metrics like loss and accuracy.

### 4. `model_builder.py`

Functions for creating various model architectures:
- `create_ResNet50`: Creates a ResNet-50 model.
- `create_ResNet18`: Creates a ResNet-18 model.
- `create_wide_resnet101_2`: Creates a Wide ResNet-101-2 model.
- `create_AlexNet`: Creates an AlexNet model.
- `create_efficientnet_b0`: Creates an EfficientNet-B0 model.
- `create_effnetb2`: Creates an EfficientNet-B2 model.
- `create_EfficientNet_V2`: Creates an EfficientNet-V2 model.
- `create_MobileNet`: Creates a MobileNet model.

### 5. `predictions.py`

Contains functions for making predictions and visualizing them:
- `pred_and_plot_image`: Predicts an image using a trained model and plots the result.

## Installation

To run this project, ensure that you have Python 3.8 or later installed. You'll also need to install the required dependencies:

```bash
pip install -r requirements.txt
```

### Dependencies

- PyTorch
- torchvision
- pandas
- matplotlib
- Pillow

## Usage

1. **Data Preparation**:
   - Prepare the dataset using `image_from_Excel_to_folder`.
   - Organize the data into training, validation, and test directories.

2. **DataLoader Creation**:
   - Use `create_dataloaders` to generate DataLoader objects for training, validation, and testing.

3. **Model Training**:
   - Create a model using one of the functions from `model_builder.py` (e.g., `create_ResNet50`).
   - Train the model using the `train` function in `engine.py`.

4. **Model Prediction**:
   - Use the `pred_and_plot_image` function to make predictions on images.

5. **Saving and Loading Models**:
   - Save your trained models with `save_model`.

## Model Architecture

This project supports various well-known architectures, including:

- **ResNet**: Used for deep residual networks. Both ResNet-50 and ResNet-18 models are supported.
- **Wide ResNet**: A wider variant of ResNet that often yields better results for image classification tasks.
- **AlexNet**: A pioneering convolutional neural network that was one of the first to perform well on large image classification tasks.
- **EfficientNet**: A family of models designed to be more efficient in terms of accuracy and model size.
- **MobileNet**: A lightweight model that is highly efficient for mobile and edge device use cases.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

