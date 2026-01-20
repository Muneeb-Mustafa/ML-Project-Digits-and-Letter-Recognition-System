import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageDraw, ImageOps
import numpy as np
import tensorflow as tf
import json
import os

class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Offline Letter & Number Recognition")
        self.root.geometry("600x500")
        self.root.configure(bg="#f0f0f0")

        # Variables
        self.model = None
        self.label_map = None
        self.pen_color = 'black'
        self.pen_size = 15 # Thicker pen for better recognition after resizing
        
        # UI Setup
        self.setup_ui()
        
        # Load Model
        self.load_model_and_labels()

    def setup_ui(self):
        # Title
        title_label = tk.Label(self.root, text="Draw a Digit or Letter", font=("Helvetica", 20, "bold"), bg="#f0f0f0")
        title_label.pack(pady=10)

        # Canvas Frame
        canvas_frame = tk.Frame(self.root, bg="white", bd=2, relief="groove")
        canvas_frame.pack(pady=10)

        # Drawing Canvas
        self.canvas_width = 300
        self.canvas_height = 300
        self.canvas = tk.Canvas(canvas_frame, width=self.canvas_width, height=self.canvas_height, bg="white", cursor="cross")
        self.canvas.pack()

        # PIL Image to draw on (in memory)
        self.image = Image.new("L", (self.canvas_width, self.canvas_height), 255) # 'L' mode = Grayscale, 255 = White
        self.draw = ImageDraw.Draw(self.image)

        # Bind events
        self.canvas.bind("<B1-Motion>", self.paint)

        # Buttons Frame
        btn_frame = tk.Frame(self.root, bg="#f0f0f0")
        btn_frame.pack(pady=20)

        # Predict Button
        predict_btn = tk.Button(btn_frame, text="Predict", command=self.predict_character, font=("Helvetica", 14), bg="#4CAF50", fg="white", width=10)
        predict_btn.pack(side="left", padx=10)

        # Clear Button
        clear_btn = tk.Button(btn_frame, text="Clear", command=self.clear_canvas, font=("Helvetica", 14), bg="#f44336", fg="white", width=10)
        clear_btn.pack(side="left", padx=10)

        # Result Label
        self.result_label = tk.Label(self.root, text="Prediction: None", font=("Helvetica", 18, "bold"), bg="#f0f0f0", fg="#333")
        self.result_label.pack(pady=10)

    def load_model_and_labels(self):
        try:
            model_path = 'model/emnist_model.h5'
            map_path = 'model/label_map.json'
            
            if not os.path.exists(model_path):
                messagebox.showwarning("Model Not Found", "Model file not found. Please run 'train_model.py' first.")
                return

            self.model = tf.keras.models.load_model(model_path)
            
            with open(map_path, 'r') as f:
                # Keys are stored as strings in JSON, need to convert back to int to lookup if we were doing reverse
                # But here we just used the prediction index to get value
                self.label_map = json.load(f)
                
            print("Model and label map loaded successfully.")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load model: {str(e)}")

    def paint(self, event):
        x1, y1 = (event.x - 1), (event.y - 1)
        x2, y2 = (event.x + 1), (event.y + 1)
        
        # Draw on Canvas
        self.canvas.create_oval(x1, y1, x2, y2, fill=self.pen_color, width=self.pen_size)
        
        # Draw on PIL Image
        # Canvas is white, we draw black.
        self.draw.line([x1, y1, x2, y2], fill=0, width=self.pen_size) # 0 = Black

    def clear_canvas(self):
        self.canvas.delete("all")
        self.image = Image.new("L", (self.canvas_width, self.canvas_height), 255)
        self.draw = ImageDraw.Draw(self.image)
        self.result_label.config(text="Prediction: None")

    def preprocess_image(self):
        # 1. Invert image (EMNIST is white on black, we drew black on white)
        # 0 is black, 255 is white.
        # We want background to be 0 (black) and digit to be 255 (white).
        img_inverted = ImageOps.invert(self.image)
        
        # 2. Get bounding box to crop empty space
        bbox = img_inverted.getbbox()
        if bbox:
            img_cropped = img_inverted.crop(bbox)
        else:
            return None # Canvas is empty

        # 3. Resize maintaining aspect ratio to fit in 20x20 box (leaving padding)
        # EMNIST/MNIST is 28x28 but the digit is centered in 20x20
        width, height = img_cropped.size
        if width > height:
            new_width = 20
            new_height = int(height * (20 / width))
        else:
            new_height = 20
            new_width = int(width * (20 / height))
            
        img_resized = img_cropped.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # 4. Paste into 28x28 black image (Center of Mass centering would be better but simple centering is okay)
        final_img = Image.new("L", (28, 28), 0)
        
        # Calculate centering position
        paste_x = (28 - new_width) // 2
        paste_y = (28 - new_height) // 2
        
        final_img.paste(img_resized, (paste_x, paste_y))
        
        # 5. Convert to numpy array and normalize
        img_array = np.array(final_img)
        img_array = img_array.astype('float32') / 255.0
        
        # 6. Reshape for model (1, 28, 28, 1)
        img_array = img_array.reshape(1, 28, 28, 1)
        
        return img_array

    def predict_character(self):
        if not self.model:
            messagebox.showerror("Error", "Model not loaded!")
            return

        input_data = self.preprocess_image()
        
        if input_data is None:
            self.result_label.config(text="Draw something first!")
            return

        # Predict
        prediction = self.model.predict(input_data)
        predicted_index = np.argmax(prediction)
        confidence = np.max(prediction)
        
        # Map index to character
        predicted_char = self.label_map.get(str(predicted_index), "?")
        
        self.result_label.config(text=f"Prediction: {predicted_char} ({confidence*100:.1f}%)")

if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()
