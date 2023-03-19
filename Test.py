import tkinter as tk
from tkinter import filedialog
from PIL import ImageTk, Image
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from tensorflow.keras.applications.resnet_v2 import preprocess_input
from tensorflow.keras.models import load_model


def load_and_prep_image(filename, img_shape_x=200, img_shape_y=200):
    img = cv2.imread(filename)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (img_shape_x, img_shape_y))
    img = preprocess_input(img)
    return img


def pred_and_plot(filename, model, class_names, img_shape_x=200, img_shape_y=200):
    img = load_and_prep_image(filename, img_shape_x, img_shape_y)
    image = io.imread(filename)
    gray_image = np.mean(image, axis=2)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    ax1.imshow(gray_image, cmap='hot')
    ax1.set_title('Heat Map')
    ax1.set_axis_off()

    pred = model.predict(np.expand_dims(img, axis=0))
    pred_class = class_names[pred.argmax()]
    img = cv2.cvtColor(cv2.imread(filename), cv2.COLOR_BGR2RGB)
    ax2.imshow(img)
    ax2.set_title(f"Prediction: {pred_class}")
    ax2.set_axis_off()

    plt.show()



def select_image():
    global panel, image_path
    image_path = filedialog.askopenfilename()
    image = Image.open(image_path)
    image = image.resize((300, 300))
    image = ImageTk.PhotoImage(image)
    if panel is None:
        panel = tk.Label(image=image)
        panel.image = image
        panel.pack(padx=10, pady=10)
    else:
        panel.configure(image=image)
        panel.image = image


def select_class_dir():
    class_dir = filedialog.askdirectory()
    class_dir_entry.delete(0, tk.END)
    class_dir_entry.insert(0, class_dir)


def select_model_file():
    model_file = filedialog.askopenfilename()
    model_file_entry.delete(0, tk.END)
    model_file_entry.insert(0, model_file)


def predict():
    global image_path, model, classes, img_shape_x, img_shape_y
    if image_path is None:
        return
    class_dir = class_dir_entry.get()
    model_file = model_file_entry.get()
    img_shape_x = int(img_shape_x_entry.get())
    img_shape_y = int(img_shape_y_entry.get())
    classes = []
    for file in os.listdir(class_dir):
        d = os.path.join(class_dir, file)
        if os.path.isdir(d):
            classes.append(file)
    model = load_model(model_file)
    pred_and_plot(image_path, model, classes, img_shape_x, img_shape_y)

root = tk.Tk()
root.title("Image Predictor")
root.resizable(False, False)

panel = None

image_path = None
model = None
classes = []
img_shape_x = 200
img_shape_y = 200

class_dir_label = tk.Label(root, text="Class Directory:")
class_dir_label.grid(row=0, column=0, padx=10, pady=10)
class_dir_entry = tk.Entry(root)
class_dir_entry.grid(row=0, column=1, padx=10, pady=10)
class_dir_button = tk.Button(root, text="Browse", command=select_class_dir)
class_dir_button.grid(row=0, column=2, padx=10, pady=10)

model_file_label = tk.Label(root, text="Model File Path:")
model_file_label.grid(row=1, column=0, padx=10, pady=10)
model_file_entry = tk.Entry(root)
model_file_entry.grid(row=1, column=1, padx=10, pady=10)
model_file_button = tk.Button(root, text="Browse", command=select_model_file)
model_file_button.grid(row=1, column=2, padx=10, pady=10)

img_shape_x_label = tk.Label(root, text="Image Shape X:")
img_shape_x_label.grid(row=2, column=0, padx=10, pady=10)
img_shape_x_entry = tk.Entry(root)
img_shape_x_entry.grid(row=2, column=1, padx=10, pady=10)

img_shape_y_label = tk.Label(root, text="Image Shape Y:")
img_shape_y_label.grid(row=3, column=0, padx=10, pady=10)
img_shape_y_entry = tk.Entry(root)
img_shape_y_entry.grid(row=3, column=1, padx=10, pady=10)

select_button = tk.Button(root, text="Select Image", command=select_image)
select_button.grid(row=4, column=0, padx=10, pady=10)

predict_button = tk.Button(root, text="Predict", command=predict)
predict_button.grid(row=4, column=1, padx=10, pady=10)

root.mainloop()
