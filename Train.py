import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
import matplotlib.pyplot as plt
plt.style.use('default')
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *

root = tk.Tk()
root.title("Skin Cancer Classification")
root.resizable(False, False)

train_dir_var = tk.StringVar()
val_dir_var = tk.StringVar()
img_shape_x_var = tk.IntVar()
img_shape_y_var = tk.IntVar()
batch_size_var = tk.IntVar()
save_model_path_var = tk.StringVar()
save_model_name_var = tk.StringVar()
num_epoch_var = tk.IntVar()

def browse_folder(var):
    folder_selected = filedialog.askdirectory()
    var.set(folder_selected)

train_dir_label = tk.Label(root, text="Training Directory: ")
train_dir_label.grid(row=0, column=0, padx=5, pady=5, sticky="w")

train_dir_entry = tk.Entry(root, textvariable=train_dir_var)
train_dir_entry.grid(row=0, column=1, padx=5, pady=5)

train_dir_button = tk.Button(root, text="Browse", command=lambda: browse_folder(train_dir_var))
train_dir_button.grid(row=0, column=2, padx=5, pady=5)

val_dir_label = tk.Label(root, text="Validation Directory: ")
val_dir_label.grid(row=1, column=0, padx=5, pady=5, sticky="w")

val_dir_entry = tk.Entry(root, textvariable=val_dir_var)
val_dir_entry.grid(row=1, column=1, padx=5, pady=5)

val_dir_button = tk.Button(root, text="Browse", command=lambda: browse_folder(val_dir_var))
val_dir_button.grid(row=1, column=2, padx=5, pady=5)

img_shape_x_label = tk.Label(root, text="Image Shape X: ")
img_shape_x_label.grid(row=2, column=0, padx=5, pady=5, sticky="w")

img_shape_x_entry = tk.Entry(root, textvariable=img_shape_x_var)
img_shape_x_entry.grid(row=2, column=1, padx=5, pady=5)

img_shape_y_label = tk.Label(root, text="Image Shape Y: ")
img_shape_y_label.grid(row=3, column=0, padx=5, pady=5, sticky="w")

img_shape_y_entry = tk.Entry(root, textvariable=img_shape_y_var)
img_shape_y_entry.grid(row=3, column=1, padx=5, pady=5)

batch_size_label = tk.Label(root, text="Batch Size: ")
batch_size_label.grid(row=4, column=0, padx=5, pady=5, sticky="w")

batch_size_entry = tk.Entry(root, textvariable=batch_size_var)
batch_size_entry.grid(row=4, column=1, padx=5, pady=5)

save_model_path_label = tk.Label(root, text="Save Model Path: ")
save_model_path_label.grid(row=5, column=0, padx=5, pady=5, sticky="w")

save_model_path_entry = tk.Entry(root, textvariable=save_model_path_var)
save_model_path_entry.grid(row=5, column=1, padx=5, pady=5)

save_model_path_button = tk.Button(root, text="Browse", command=lambda: browse_folder(save_model_path_var))
save_model_path_button.grid(row=5, column=2, padx=5, pady=5)

save_model_name_label = tk.Label(root, text="Save Model Name: ")
save_model_name_label.grid(row=6, column=0, padx=5, pady=5, sticky="w")

save_model_name_entry = tk.Entry(root, textvariable=save_model_name_var)
save_model_name_entry.grid(row=6, column=1, padx=5, pady=5)

num_epoch_label = tk.Label(root, text="Number of Epochs: ")
num_epoch_label.grid(row=7, column=0, padx=5, pady=5, sticky="w")

num_epoch_entry = tk.Entry(root, textvariable=num_epoch_var)
num_epoch_entry.grid(row=7, column=1, padx=5, pady=5)

def Create_ResNet50V2_Model(num_classes, img_shape_x, img_shape_y):
    ResNet50V2 = tf.keras.applications.ResNet50V2(input_shape=(img_shape_x, img_shape_y, 3),
    include_top= False,
    weights='imagenet'
    )
    ResNet50V2.trainable = True

    for layer in ResNet50V2.layers[:-50]:
        layer.trainable = False


    model = Sequential([
                    ResNet50V2,
                    Dropout(.25),
                    BatchNormalization(),
                    Flatten(),
                    Dense(64, activation='relu'),
                    BatchNormalization(),
                    Dropout(.5),
                    Dense(num_classes,activation='softmax')
                    ])
    return model

def train_model():
    train_data_path = train_dir_var.get()
    val_data_path = val_dir_var.get()
    img_shape_x = img_shape_x_var.get()
    img_shape_y = img_shape_y_var.get()
    batch_size = batch_size_var.get()
    save_model_path = save_model_path_var.get()
    save_model_name = save_model_name_var.get()
    num_epoch = num_epoch_var.get()

    train_preprocessor = ImageDataGenerator(
        rescale = 1 / 255.,
        rotation_range=10,
        zoom_range=0.2,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,                                        
        fill_mode='nearest',
    )

    val_preprocessor = ImageDataGenerator(
        rescale = 1 / 255.,
    )

    train_data = train_preprocessor.flow_from_directory(
        train_data_path,
        class_mode="categorical",
        target_size=(img_shape_x,img_shape_y),
        color_mode='rgb', 
        shuffle=True,
        batch_size=batch_size,
        subset='training', 
    )

    val_data = val_preprocessor.flow_from_directory(
        val_data_path,
        class_mode="categorical",
        target_size=(img_shape_x,img_shape_y),
        color_mode="rgb",
        shuffle=False,
        batch_size=batch_size,
    )

    num_classes = train_data.num_classes

    ResNet50V2_Model = Create_ResNet50V2_Model(num_classes, img_shape_x, img_shape_y)
    ResNet50V2_Model.summary()

    ResNet50V2_Model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    checkpoint_path = os.path.join(save_model_path, save_model_name)
    ModelCheckpoint(checkpoint_path, monitor="val_accuracy", save_best_only=True)

    Early_Stopping = EarlyStopping(monitor = 'val_accuracy', patience = 7, restore_best_weights = True, verbose=1)

    Reducing_LR = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
                                                factor=0.2,
                                                patience=2,
                                                verbose=1)

    callbacks = [Early_Stopping, Reducing_LR]

    steps_per_epoch = train_data.n // train_data.batch_size
    validation_steps = val_data.n // val_data.batch_size

    ResNet50V2_history = ResNet50V2_Model.fit(train_data ,validation_data = val_data , epochs=num_epoch, batch_size=batch_size,
                                        callbacks = callbacks, steps_per_epoch=steps_per_epoch, validation_steps=validation_steps)
    ResNet50V2_Model.save(os.path.join(save_model_path, save_model_name + '.h5'))
    def plot_curves(history):

        loss = history.history["loss"]
        val_loss = history.history["val_loss"]

        accuracy = history.history["accuracy"]
        val_accuracy = history.history["val_accuracy"]

        epochs = range(len(history.history["loss"]))

        plt.figure(figsize=(15,5))

        plt.subplot(1, 2, 1)
        plt.plot(epochs, loss, label = "training_loss")
        plt.plot(epochs, val_loss, label = "val_loss")
        plt.title("Loss")
        plt.xlabel("epochs")
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(epochs, accuracy, label = "training_accuracy")
        plt.plot(epochs, val_accuracy, label = "val_accuracy")
        plt.title("Accuracy")
        plt.xlabel("epochs")
        plt.legend()

        plt.show()
        
    plot_curves(ResNet50V2_history)
    messagebox.showinfo(title="Training Completed", message="The model has been trained successfully!")

train_button = tk.Button(root, text="Train", command=train_model)
train_button.grid(row=8, column=1, padx=5, pady=5)

root.mainloop()