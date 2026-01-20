# Offline Letter & Number Recognition System

A Python desktop application that uses a Convolutional Neural Network (CNN) to recognize handwritten digits (0-9) and letters (A-Z, a-z) drawn by the user.

## Project Structure
- `app.py`: The main GUI application where you can draw and predict characters.
- `train_model.py`: Script to train the CNN model using the EMNIST dataset.
- `model/`: Directory where the trained model (`emnist_model.h5`) and label mapping (`label_map.json`) are saved.
- `requirements.txt`: List of required Python libraries.

## Prerequisites
- Python 3.8 or higher installed.

## Setup Instructions

1.  **Open Command Prompt (Terminal)** in the project folder.

2.  **Install dependencies**:
    Run the following command to install necessary libraries:
    ```bash
    pip install -r requirements.txt
    ```

## How to Run

### Step 1: Train the Model (First Time Only)
Before running the app, you need to train the Machine Learning model. This requires an internet connection *initially* to download the EMNIST dataset (approx 700MB). Once downloaded, it will work offline.

Run:
```bash
python train_model.py
```
*Note: This process may take a few minutes depending on your computer speed. It will train for 5 epochs and save the model to the `model` folder.*

### Step 2: Run the Application
Once the model is trained, launch the desktop app:

```bash
python app.py
```

### Step 3: Use the App
1.  Draw a character (0-9, A-Z, or a-z) in the write box using your mouse.
2.  Click **Predict**.
3.  The recognized character will appear below.
4.  Click **Clear** to start over.

## Troubleshooting
- **dataset not found**: If `train_model.py` fails to download/find the dataset, ensure you have internet access.
- **Model not found in app**: Make sure you ran `train_model.py` successfully and the `model` folder contains `emnist_model.h5`.

## About
Built for Machine Learning Project. Uses TensorFlow/Keras and Tkinter.
