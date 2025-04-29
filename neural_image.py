# Neural Networking
import tensorflow as tf
import numpy as np
from siren_model import Sine, build_siren_model, ModelConfig

# video recording & file saving
import cv2
import os

from video_callback import VideoCallback

# nice printing colors
from colorama import Fore, Style

# users window size
import tkinter as tk


print(Fore.CYAN + f'TensorFlow Version: {tf.__version__}' + Style.RESET_ALL)

""" Settings """
epochs = 50000
image_path = "images/bee.png"

aspect_ratio = 1024/683 #1920 / 1080
height = 256
image_size = (int(height * aspect_ratio), height)

hidden_layers = 6
neurons_per_layer = 175

weights_save_path = "outputs/network_data.weights.h5"
video_save_path = "outputs/training_video.mp4"
video_frame_rate = 144

""" ~ ~ ~ ~ """

class LossHistory(tf.keras.callbacks.Callback):
    def __init__(self):
        super().__init__()
        self.losses = []

    def on_epoch_end(self, epoch, logs=None):
        if logs is not None and 'loss' in logs:
            self.losses.append(logs['loss'])


def load_image_from_file(image_path : str, desired_size: tuple) -> np.ndarray:
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return cv2.resize(img, dsize=desired_size, interpolation=cv2.INTER_CUBIC)


def get_window_dims():
    root = tk.Tk()
    width = root.winfo_screenwidth()
    height = root.winfo_screenheight()
    root.destroy()
    return width, height


class NeuralImageGenerator:
    def __init__(self, load_model = False):
        # loading the image with our desired shape and resolution
        self.image = load_image_from_file(image_path, image_size)

        # getting the input data and its corresponding desired output data
        self.input_data = self.create_input_data(image_size)
        self.target_output = self.create_output_data()

        # creating the model we use for training
        config = ModelConfig(
            input_dim=7,
            hidden_layers=14,
            hidden_units=256,
            w0=1.0,
            w0_initial=30.0,
            final_activation='sigmoid'
        )

        self.model = build_siren_model(config)
        if load_model:
            self.load_model()
        self.model.summary()
        
        # for recording loss over time
        self.loss_callback = LossHistory()
        self.video_callback = VideoCallback(self, save_every=1, resolution=image_size, output_path=video_save_path)
    

    def train_model(self, save_model = True, hyper_res=False) -> tuple:
         # Create a callback that saves the model's weights
        cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=weights_save_path, save_weights_only=True, verbose=1, mode='max', save_best_only='true')

        # Create tf.data.Dataset pipeline
        dataset = tf.data.Dataset.from_tensor_slices((self.input_data, self.target_output))
        #dataset = dataset.batch(batch_size).cache().prefetch(tf.data.AUTOTUNE)

        dataset = dataset.batch(self.input_data.shape[0])  # Whole image in one batch
        
        try: 
            self.model.compile(optimizer='adam', loss='mse')
            self.model.fit(
                dataset, 
                epochs=epochs,
                callbacks=[self.loss_callback, cp_callback, self.video_callback]
            )
        except KeyboardInterrupt:
            print(Fore.RED + "\nTraining stopped. Compiling Results. . ." + Style.RESET_ALL)
        
        print("==== Saving Video ==== ")
        self.video_callback.save_video(fps=video_frame_rate)

        print("==== Evaluating model ==== ")
        evalutation = self.model.evaluate(self.input_data, self.target_output, verbose=2)
        
        prediction, size = self.get_prediction(hyper_res)
    
        return prediction, evalutation, size
   
    def get_prediction(self, hyper_res=False):
        if not hyper_res:
            return self.model.predict(self.input_data), image_size

        size = (1920, 1080)
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

    def create_output_data(self):
        # the image needs to be normalized from 0-255 to 0-1 and have a 2d shape
        normalized = self.image / 255
        return normalized.reshape(image_size[0] * image_size[1], 3)

    def load_model(self):
        if not os.path.exists(weights_save_path):
            print(Fore.RED + "No previous save file exists, creating new model" + Style.RESET_ALL)
            return
        
        print(Fore.BLUE + f"A model has been retrived from '{weights_save_path}'")
        if input(Fore.BLUE + "would you like to continue training with this model? y/n " + Style.RESET_ALL) == 'y':            
            self.model.load_weights(weights_save_path)
      

    def create_model(self):
        # we create a tensorflow sequential model 7 inputs to 3 outputs
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Input((7,))) # input layer

        for _ in range(int(hidden_layers / 2)):
            model.add(tf.keras.layers.Dense(neurons_per_layer))
            model.add(Sine(w0=30))

        # last layer is a sigmoid to clamp values between 0 and 1 for coloring
        model.add(tf.keras.layers.Dense(3, activation='sigmoid')) # output layer
        
        # compiling the model
        model.compile(optimizer='adam', loss='mse')

        return model
    
