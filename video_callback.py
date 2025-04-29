# Neural Networking
import tensorflow as tf
import numpy as np

# video recording
import cv2

# nice printing colors
from colorama import Fore, Style


class VideoCallback(tf.keras.callbacks.Callback):
    def __init__(self, model_ref, save_every=10, resolution=(64, 64), output_path='training_visual.mp4'):
        super().__init__()
        self.model_ref = model_ref  # NeuralImageGenerator instance
        self.save_every = save_every
        self.resolution = resolution
        self.frames = []
        self.output_path = output_path

    def on_epoch_end_old(self, epoch, logs=None):
        if epoch % self.save_every == 0:
            print(Fore.YELLOW + f"\n[VideoCallback] Generating frame at epoch {epoch}..." + Style.RESET_ALL)
            self.generate_frame()
    
    def on_epoch_end(self, epoch, logs=None):
        if epoch % self.save_every == 0:
            print(Fore.YELLOW + f"\n[VideoCallback] Generating frame at epoch {epoch}..." + Style.RESET_ALL)
            coords = self.model_ref.input_data  # Already full-size input
            preds = self.model_ref.model(coords, training=False).numpy()
            frame = (preds.reshape((*self.resolution[::-1], 3)) * 255).astype(np.uint8)
            self.frames.append(frame)


    def on_train_end(self, logs=None):
        print(Fore.CYAN + "\n[VideoCallback] Saving final video..." + Style.RESET_ALL)
        self.save_video()

    def generate_frame(self):
        preds, size = self.model_ref.get_prediction(hyper_res=False)
        
        # Reshape into image
        frame = (preds.reshape((*self.resolution[::-1], 3)) * 255).astype(np.uint8)
        self.frames.append(frame)

    def save_video(self, fps=10):
        if not self.frames:
            print(Fore.RED + "[VideoCallback] No frames generated." + Style.RESET_ALL)
            return

        height, width = self.resolution[1], self.resolution[0]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(self.output_path, fourcc, fps, (width, height))

        for frame in self.frames:
            out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))  # Convert RGB to BGR for OpenCV
        out.release()
        print(Fore.GREEN + f"[VideoCallback] Video saved to {self.output_path}" + Style.RESET_ALL)
