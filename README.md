# Emotion Recognition CNN

A deep learning project for facial emotion recognition using Convolutional Neural Networks (CNN) built with TensorFlow/Keras.

## Overview

This project implements a CNN model to classify facial expressions into 7 different emotion categories. The model uses grayscale images of size 48x48 pixels and employs various data augmentation techniques to improve performance.

## Features

-   **Deep CNN Architecture**: 5 convolutional layers with batch normalization and dropout
-   **Data Augmentation**: Rotation, zoom, and horizontal flip for better generalization
-   **Regularization**: L2 regularization and dropout to prevent overfitting
-   **Callbacks**: Early stopping and learning rate scheduling for optimal training
-   **Visualization**: Training plots, confusion matrix, and emotion prediction display
-   **Real-time Prediction**: Function to predict emotions from new images

## Model Architecture

```
Input (48x48x1) → Conv2D(32) → Conv2D(64) → BatchNorm → MaxPool → Dropout
                → Conv2D(128) → BatchNorm → MaxPool → Dropout
                → Conv2D(512) → BatchNorm → MaxPool → Dropout (×3)
                → Flatten → Dense(256) → BatchNorm → Dropout
                → Dense(512) → BatchNorm → Dropout
                → Dense(7, softmax)
```

## Dataset Structure

```
dataset/
├── train/
│   ├── emotion1/
│   ├── emotion2/
│   └── ...
└── validation/
    ├── emotion1/
    ├── emotion2/
    └── ...
```

## Installation

1. Clone this repository:

```bash
git clone <repository-url>
cd emotion-recognition-cnn
```

2. Install required dependencies:

```bash
pip install -r requirements.txt
```

3. Ensure your dataset is organized in the correct structure as shown above.

## Usage

### Training the Model

Run the main script to train the model:

```python
python emotion_recognition.py
```

The training process includes:

-   Data loading and augmentation
-   Model compilation with Adam optimizer
-   Training with early stopping and learning rate reduction
-   Visualization of training metrics
-   Model evaluation with classification report and confusion matrix

### Making Predictions

Use the `detect_emotion()` function to predict emotions from new images:

```python
predicted_emotion, confidence = detect_emotion('path/to/your/image.jpg')
print(f"Predicted: {predicted_emotion} with {confidence}% confidence")
```

## Training Configuration

-   **Input Size**: 48x48 grayscale images
-   **Batch Size**: 64
-   **Epochs**: 20 (with early stopping)
-   **Optimizer**: Adam (learning_rate=0.01)
-   **Loss Function**: Categorical Crossentropy
-   **Data Augmentation**: Rotation (20°), Zoom (0.2), Horizontal flip

## Model Performance

The model includes several techniques to improve performance:

-   Batch normalization for stable training
-   Dropout layers (0.25) to prevent overfitting
-   L2 regularization on deeper convolutional layers
-   Early stopping to prevent overtraining
-   Learning rate reduction on plateau

## Visualization Features

-   Sample training images with emotion labels
-   Training/validation loss and accuracy curves
-   Confusion matrix heatmap
-   Individual prediction visualization with confidence scores

## Requirements

-   Python 3.7+
-   TensorFlow 2.x
-   NumPy
-   Matplotlib
-   Seaborn
-   Scikit-learn

## File Structure

```
├── emotion_recognition.py    # Main training and prediction script
├── dataset/                 # Dataset directory
├── README.md               # Project documentation
├── requirements.txt        # Python dependencies
└── .gitignore             # Git ignore file
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/new-feature`)
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

-   TensorFlow/Keras for the deep learning framework
-   The emotion recognition dataset contributors
-   Open source community for tools and libraries
