# Neural Networking
import tensorflow as tf
import numpy as np

# optimization with multithreading
import threading
import queue

# video recording
import cv2

# file saving

# nice printing colors
from colorama import Fore, Style

# users window size
import tkinter as tk


print(Fore.CYAN + f'TensorFlow Version: {tf.__version__}' + Style.RESET_ALL)

""" Settings """
epochs = 50000
image_path = "images/calculator.png"

aspect_ratio = 1920 / 1080
height = 256
image_size = (int(height * aspect_ratio), height)

batch_size = 100_000
print(Fore.GREEN + "Batch Size: " + Style.RESET_ALL, batch_size)

hidden_layers = 4
neurons_per_layer = 150
activation_func = 'tanh'

save_to_path = "network_data.pkl"

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

def load_model_from_file(image_path : str):
    #with open(image_path, 'rb') as file:
    #    loaded = pickle.load(file)
    #return loaded
    return None

def write_model_to_file(image_path : str, model):
    #with open(image_path, 'wb') as file:
    #    pickle.dump(model, file)
    pass


def get_window_dims():
    root = tk.Tk()
    width = root.winfo_screenwidth()
    height = root.winfo_screenheight()
    root.destroy()
    return width, height


class Sine(tf.keras.layers.Layer):
    def __init__(self, w0=1.0):
        super().__init__()
        self.w0 = w0

    def call(self, inputs):
        return tf.sin(self.w0 * inputs)

# Custom Weight Initializer
class SIRENInitializer(tf.keras.initializers.Initializer):
    def __init__(self, w0=1.0):
        self.w0 = w0

    def __call__(self, shape, dtype=None):
        input_dim = shape[0]
        limit = np.sqrt(6 / input_dim) / self.w0
        return tf.random.uniform(shape, minval=-limit, maxval=limit, dtype=dtype)

# Building the full model
def build_siren_model(input_dim, hidden_layers=15, hidden_units=200, w0=30.0, final_activation='sigmoid'):
    inputs = tf.keras.Input(shape=(input_dim,))

    # First layer
    x = tf.keras.layers.Dense(
        hidden_units,
        kernel_initializer=SIRENInitializer(w0=1.0),
        use_bias=False)(inputs)
    x = Sine(w0=1.0)(x)

    # Hidden layers
    for _ in range(hidden_layers):
        x = tf.keras.layers.Dense(
            hidden_units,
            kernel_initializer=SIRENInitializer(w0=w0))(x)
        x = Sine(w0=w0)(x)

    # Output layer
    outputs = tf.keras.layers.Dense(3, activation=final_activation)(x)

    model = tf.keras.Model(inputs, outputs)
    return model

class NeuralImageGenerator:
    def __init__(self, load_model = False):
        # loading the image with our desired shape and resolution
        self.image = load_image_from_file(image_path, image_size)

        # getting the input data and its corresponding desired output data
        self.input_data = self.create_input_data(image_size)
        self.target_output = self.create_output_data()

        # creating the model we use for training
        self.model = build_siren_model(7, hidden_layers, neurons_per_layer, 1)
        if load_model:
            self.load_model()
        self.model.summary()
        
        # for recording loss over time
        self.loss_callback = LossHistory()
    

    def train_model(self, save_model = True, hyper_res=False) -> tuple:
        try:
            # Create tf.data.Dataset pipeline
            dataset = tf.data.Dataset.from_tensor_slices((self.input_data, self.target_output))
            dataset = dataset.batch(batch_size).cache().prefetch(tf.data.AUTOTUNE)
            
            self.model.compile(optimizer='adam', loss='mse')
            self.model.fit(
                dataset, 
                epochs=epochs,
                callbacks=[self.loss_callback]
            )
        except KeyboardInterrupt:
            user_input = input(Fore.BLUE + "\nTraining stopped. would you like to save the current training progress? y/n " + Style.RESET_ALL)
            save_model = user_input == "y"
        
        print("==== Evaluating model ==== ")
        evalutation = self.model.evaluate(self.input_data, self.target_output, verbose=2)
        
        prediction, size = self.get_prediction(hyper_res)

        if save_model:
            write_model_to_file(save_to_path, self.model)
            print(Fore.GREEN + f"Model saved to {save_to_path}" + Style.RESET_ALL)
    
        return prediction, evalutation, size
   
    def get_prediction(self, hyper_res=False):
        size = (1920, 1080) if hyper_res else image_size
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
        model = load_model_from_file(save_to_path)
        if (model != None):
            print(Fore.BLUE + f"A model has been retrived from '{save_to_path}'")
            model.summary()
            if input(Fore.BLUE + "would you like to continue training with this model? y/n " + Style.RESET_ALL) == 'y':
                self.model = model


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
    
