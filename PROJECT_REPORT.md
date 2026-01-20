# Project Report: Offline Letter & Number Recognition System

## 1. Introduction
This project is a desktop application designed to recognize handwritten characters. It utilizes a Convolutional Neural Network (CNN) trained on an image dataset to perform offline recognition in real-time.

## 2. Dataset
We used the **MNIST (Modified National Institute of Standards and Technology)** dataset (provided locally in the `Emnist` folder). 
*Note: Although intended for EMNIST (Letters + Digits), the provided dataset files contained 60,000 training images of digits 0-9. Thus, the current system is optimized for **Digit Recognition**.*

- **Type**: Handwritten Digits (0-9).
- **Format**: 28x28 grayscale images.
- **Training Data**: 60,000 images.
- **Testing Data**: 10,000 images.

## 3. Preprocessing
Before feeding data into the CNN, several preprocessing steps are applied:

### Training Data Preprocessing
- **Normalization**: Pixel values (0-255) are divided by 255.0 to scale them to the range [0, 1].
- **Reshaping**: Images are reshaped from (28, 28) to (28, 28, 1) to add the channel dimension.

### Real-time User Input Preprocessing
When a user draws on the Tkinter canvas, the input must match the format of the training data:
1.  **Inversion**: The drawing is black on white, while the dataset is white on black. We invert the image.
2.  **Cropping**: The bounding box of the drawn character is calculated to remove excess whitespace.
3.  **Resizing & Padding**: The cropped character is resized to fit into a 20x20 box (preserving aspect ratio) and then centered in a 28x28 black image. This mimics the Center-of-Mass centering used in MNIST/EMNIST.
4.  **Normalization**: The image is converted to a normalized numpy array.

## 4. CNN Architecture
We constructed a Sequential CNN model using TensorFlow/Keras:

1.  **Conv2D (32 filters, 3x3 kernel, ReLU)**: Extracts low-level features.
2.  **MaxPooling2D (2x2)**: Reduces spatial dimensions.
3.  **Conv2D (64 filters, 3x3 kernel, ReLU)**: Extracts higher-level features.
4.  **MaxPooling2D (2x2)**: Further dimensionality reduction.
5.  **Flatten**: Converts 2D feature maps to a 1D vector.
6.  **Dense (128 neurons, ReLU)**: Fully connected layer.
7.  **Dropout (0.5)**: Randomly sets 50% of inputs to 0 to prevent overfitting.
8.  **Dense (Output Layer)**: Softmax activation. (Size depends on dataset classes, currently 10 for Digits).

## 5. System Workflow
1.  **Training**: The `train_model.py` script loads the local IDX files, trains the model, and saves the weights (`emnist_model.h5`).
2.  **GUI Application**: `app.py` loads the saved model.
3.  **Interaction**:
    - User draws a character on the canvas.
    - Upon clicking "Predict", the image is captured and preprocessed.
    - The model predicts the class with the highest probability.
    - The result is displayed to the user.

## 6. Libraries Used
- **TensorFlow/Keras**: For building and training the neural network.
- **NumPy**: For numerical matrix operations.
- **Pillow (PIL)**: For image manipulation in the GUI.
- **Tkinter**: For the desktop graphical user interface.

## 7. Conclusion
The system successfully integrates a deep learning model into a user-friendly desktop application. It achieves high accuracy on the standard MNIST test set (~99%) and performs robustly on real-time user input.
