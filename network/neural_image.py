# Neural Networking
import tensorflow as tf
import numpy as np
from network.siren_model import Sine, build_siren_model, ModelConfig

# video recording & file saving
import cv2
import os

from network.video_callback import VideoCallback

# nice printing colors
from colorama import Fore, Style

# users window size
import tkinter as tk



import sys
class SingleLineLogger(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        msg = f"\rEpoch {epoch+1} | " + " | ".join([f"{k}: {v:.4f}" for k, v in logs.items()])
        sys.stdout.write(msg)
        sys.stdout.flush()

    def on_train_end(self, logs=None):
        print()  # move to new line after training



print(Fore.CYAN + f'TensorFlow Version: {tf.__version__}' + Style.RESET_ALL)

""" Settings """
epochs_per_image = 1500
video_frame_rate = 30

height = 256 #256


 # creating the model we use for training
config = ModelConfig(
    input_dim=7,
    hidden_layers=15,
    hidden_units=180,
    w0=1.0,
    w0_initial=30.0,
    final_activation='sigmoid'
)

training_image_folder = "training_images"
weights_save_path = "outputs/network_data.weights.h5"
video_save_path = "outputs/training_videos/"
final_image_save_path = "outputs/final_images/"

""" ~ ~ ~ ~ """

class LossHistory(tf.keras.callbacks.Callback):
    def __init__(self):
        super().__init__()
        self.losses = []

    def on_epoch_end(self, epoch, logs=None):
        if logs is not None and 'loss' in logs:
            self.losses.append(logs['loss'])


def load_image_from_file(image_path: str, desired_shortest_side: int) -> np.ndarray:
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Image at {image_path} could not be read.")
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    h, w = img.shape[:2]
    if h < w:
        scale = desired_shortest_side / h
    else:
        scale = desired_shortest_side / w

    new_size = (int(w * scale), int(h * scale))
    return cv2.resize(img, new_size, interpolation=cv2.INTER_CUBIC)

def load_all_images_from_folder(folder_path: str, desired_shortest_side: int) -> list:
    supported_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp'}
    images = []
    file_names = os.listdir(folder_path)

    for filename in file_names:
        if any(filename.lower().endswith(ext) for ext in supported_extensions):
            image_path = os.path.join(folder_path, filename)
            try:
                img = load_image_from_file(image_path, desired_shortest_side)
                images.append(img)
            except Exception as e:
                print(f"Skipping {filename}: {e}")

    file_names_no_ext = [os.path.splitext(name)[0] for name in file_names] # removes the .png, .jpg etc
    return images, file_names_no_ext


def get_window_dims():
    root = tk.Tk()
    width = root.winfo_screenwidth()
    height = root.winfo_screenheight()
    root.destroy()
    return width, height


class NeuralImageGenerator:
    def __init__(self, load_model = False):
        # loading the image with our desired shape and resolution
        self.images, self.image_names = load_all_images_from_folder(training_image_folder, height)
        self.image_sizes = [(img.shape[1], img.shape[0]) for img in self.images]

        # creating the input data from the image sizes
        self.images_inputs = [self.create_input_data(size) for size in self.image_sizes]

        # now we need to turn these images into data that the network can handle
        normalized_images = [self.normalize_reshape_image(img, size) for img, size in zip(self.images, self.image_sizes)]
        self.datasets = [self.create_dataset(inp, norm) for inp, norm in zip(self.images_inputs, normalized_images)]

        # loading the model
        self.model = build_siren_model(config)
        if load_model:
            self.load_model()
        self.model.summary()

        self.model.compile(optimizer='adam', loss='mse')

        self.preditions = []
        
        # for recording loss over time
        self.loss_callback = LossHistory()
        self.video_callback = VideoCallback(self, save_every=1, resolution=self.image_sizes[0])
    

    def train_model(self, save_model = True, hyper_res=False) -> tuple:
        # Create a callback that saves the model's weights
        cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=weights_save_path, save_weights_only=True, verbose=1, mode='max', save_best_only=True)
        callbacks = [self.loss_callback, cp_callback, self.video_callback, SingleLineLogger()]
        
        # each model is trained one after the other
        image_index = 0
        for dataset, name, size in zip(self.datasets, self.image_names, self.image_sizes):
            # setting the video maker
            self.video_callback.reset(size, image_index)

            exeption = self.fit_image(dataset, callbacks, name)
            
            print("==== Saving Video ==== ")
            path = video_save_path + name + ".mp4"
            print(Fore.BLUE + f"saving to path '{path}'" + Style.RESET_ALL)

            self.video_callback.save_video(fps=video_frame_rate, output_path=path)
            

            print("==== Evaluating model ==== ")
            self.preditions.append(self.get_prediction(image_index, hyper_res)[0])
            
            if exeption:
                break
            image_index += 1
        
        return self.preditions, self.image_sizes, self.images


    def fit_image(self, dataset, call_backs, name):
        # training this image
        exeption = False # if the user interupts we want to stop the program
        try: 
            self.model.fit(
                    dataset, 
                    epochs=epochs_per_image,
                    callbacks=call_backs,
                    verbose=0)
        except KeyboardInterrupt: 
            print(Fore.RED + "\nTraining stopped. Compiling Results. . ." + Style.RESET_ALL)
            exeption = True

        print(Fore.MAGENTA + f"Training finished for {name}" + Style.RESET_ALL)
        return exeption
            
        
   
    def get_prediction(self, index, hyper_res=False) -> tuple[np.ndarray, tuple]:
        original_size = self.image_sizes[index]
        inp_data = self.images_inputs[index]

        if not hyper_res:
            return self.model.predict(inp_data), original_size

        size = original_size * 4
        input_space = self.create_input_data(size)
        return self.model.predict(input_space), size

    def get_losses(self):
        return self.loss_callback.losses

    def create_input_data(self, size):        
        # pre-creating the input space
        sx, sy = size
        x, y = np.meshgrid(np.linspace(0, 1, sx), np.linspace(0, 1, sy), indexing='xy')

        # Cartesian Top-Left & Bottom-Right
        input_space_TL = np.column_stack((x.ravel(), y.ravel()))
        input_space_BR = np.column_stack(((1 - x).ravel(), (1 - y).ravel()))

        # Polar (centered)
        dx, dy = x - 0.5, y - 0.5
        r = np.sqrt(dx**2 + dy**2)
        theta = np.arctan2(dy, dx)
        polar_space = np.column_stack((r.ravel(), np.sin(theta).ravel(), np.cos(theta).ravel()))

        # [TL.x, TL.y, BR.x, BR.y, R, Sin, Cos]
        return np.concatenate((input_space_TL, input_space_BR, polar_space), axis=1)

    def normalize_reshape_image(self, image, size):
        # the image needs to be normalized from 0-255 to 0-1 and have a 2d shape
        normalized = image / 255
        return normalized.reshape(size[0] * size[1], 3)

    def create_dataset(self, inp_data, normalised_img):
        dataset = tf.data.Dataset.from_tensor_slices((inp_data, normalised_img))
        dataset = dataset.batch(inp_data.shape[0])
        return dataset


    def load_model(self):
        if not os.path.exists(weights_save_path):
            print(Fore.RED + "No previous save file exists, creating new model" + Style.RESET_ALL)
            return
        
        print(Fore.BLUE + f"A model has been retrived from '{weights_save_path}'")
        if input(Fore.BLUE + "would you like to continue training with this model? y/n " + Style.RESET_ALL) == 'y':            
            self.model.load_weights(weights_save_path)
      
    
