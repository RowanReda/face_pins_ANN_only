
### Pins Face Recognition - Multiclass Classification with Keras
## Objective
The goal of this project is to build a multiclass classification model using Keras to recognize and classify images of faces based on the identity of different celebrities. The target accuracy for the model is set at 85% or higher on the validation or test dataset.

## Dataset
  *Dataset Link: 
    Pins Face Recognition Dataset on Kaggle
  # -Description: 
    The dataset contains facial images categorized by individual celebrities or persons, where each class represents a unique identity.

## Requirements
  # This project uses Python and requires the following main packages:
    tensorflow (with Keras)
    mtcnn
    lz4 and joblib for optimized data loading
    cv2 (OpenCV) for image preprocessing

## Steps
  # 1. Data Loading and Preprocessing
  -Resizing: Images are resized to 100x100 pixels for uniform input dimensions.

  -Normalization: Pixel values are scaled between 0 and 1.

  -Splitting: Dataset is split into 80% training and 20% validation sets.

  -Data Augmentation: Random flips, zooms, and rotations are applied to enhance generalization.

  - Face Detection: MTCNN is used to detect and crop faces, ensuring better focus on facial features.
    
## 2. Model Building (Artificial Neural Network)
The model is built using Keras Sequential API:
  -Input Layer: Accepts 100x100 resized images (flattened to 10,000 features).
  -Hidden Layers: Dense layers with ReLU activation (512, 256, and 128 neurons).
  -Dropout: Used in hidden layers to prevent overfitting.
  -Output Layer: Softmax layer with neurons equal to the number of classes for multiclass classification.
  -Early Stopping & Learning Rate Scheduler: Applied to optimize training and prevent overfitting.

## 3. Training
  The model is trained using the training set with a validation split of 20%. The primary evaluation metric is accuracy.

## 4. Evaluation
  -Classification Report: Precision, recall, and F1-score are calculated.
  -Loss and Accuracy Curves: Plots for training and validation losses and accuracies across epochs.
  -Confusion Matrix: Visual representation of classification performance across classes.

Results
Include any insights gained from the model’s performance, hyperparameter tuning, and challenges faced.

Visualizations
Training & Validation Curves: Plot loss and accuracy over epochs.
Confusion Matrix: Display the true vs. predicted classes.
Sample Images: Display examples of correctly and incorrectly classified images.
