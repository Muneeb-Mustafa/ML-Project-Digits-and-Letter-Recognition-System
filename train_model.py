import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
import emnist
import json
import os

# Create a project directory for models if it doesn't exist
if not os.path.exists('model'):
    os.makedirs('model')

print("Loading EMNIST 'byclass' dataset...")
print("If this fails, please run 'python download_data.py' first.")

try:
    # 'byclass' split has 62 classes: 0-9, A-Z, a-z
    # Images are 28x28
    x_train, y_train = emnist.extract_training_samples('byclass')
    x_test, y_test = emnist.extract_test_samples('byclass')
except Exception as e:
    print(f"\nError loading dataset: {e}")
    print("Please run 'python download_data.py' to fix the dataset.")
    exit(1)

print(f"Training data shape: {x_train.shape}")
print(f"Test data shape: {x_test.shape}")

# Use tf.data.Dataset to avoid MemoryError
# We keep x_train/x_test as uint8 (small) and convert to float32 (large) only in batches during training.

def preprocess_image(image, label):
    # Normalize: Convert uint8 to float32 and divide by 255.0
    image = tf.cast(image, tf.float32) / 255.0
    # Reshape: Add channel dimension (28, 28) -> (28, 28, 1)
    image = tf.expand_dims(image, -1)
    return image, label

# Create datasets from the numpy arrays
batch_size = 128

train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
# Shuffle, map preprocessing, batch, and prefetch for performance
train_ds = train_ds.shuffle(10000).map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE).batch(batch_size).prefetch(tf.data.AUTOTUNE)

test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test))
test_ds = test_ds.map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE).batch(batch_size).prefetch(tf.data.AUTOTUNE)

# Free up the raw numpy memory now that the dataset references it (or to help GC if copies were made)
# Note: from_tensor_slices usually creates a copy or reference. 
# Deleting the original variables helps if deep copies weren't made or just to be clean.
del x_train, y_train, x_test, y_test
import gc
gc.collect()

print("Data pipline set up. Memory optimized.")

# Defined number of classes
num_classes = 62

# Create the CNN Model
model = models.Sequential([
    # First Convolutional Block
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    
    # Second Convolutional Block
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    # Flattening
    layers.Flatten(),
    
    # Dense Layers
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5), # Reduces overfitting
    layers.Dense(num_classes, activation='softmax') # Output layer
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy', # Use sparse since labels are integers
              metrics=['accuracy'])

print("\nModel Architecture:")
model.summary()

# Train the Model
print("\nStarting Training... (This might take a few minutes)")
# Pass the dataset objects instead of numpy arrays
history = model.fit(train_ds, epochs=5, validation_data=test_ds)

# Save the Model
model_path = 'model/emnist_model.h5'
model.save(model_path)
print(f"\nModel saved to {model_path}")

# Evaluate
print("Evaluating model...")
loss, acc = model.evaluate(test_ds)
print(f"Test Accuracy: {acc*100:.2f}%")

# Create and Save Label Mapping
# EMNIST ByClass Mapping:
# 0-9: Digits
# 10-35: Uppercase A-Z
# 36-61: Lowercase a-z
label_map = {}

# Digits
for i in range(10):
    label_map[i] = str(i)

# Uppercase
for i in range(26):
    label_map[i + 10] = chr(ord('A') + i)

# Lowercase
for i in range(26):
    label_map[i + 36] = chr(ord('a') + i)

with open('model/label_map.json', 'w') as f:
    json.dump(label_map, f)
print("Label map saved to model/label_map.json")
