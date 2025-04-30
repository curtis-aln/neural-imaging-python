# Neural Networking
import tensorflow as tf
import numpy as np

# video recording
import cv2
import os

# nice printing colors
from colorama import Fore, Style


class VideoCallback(tf.keras.callbacks.Callback):
    def __init__(self, model_ref, save_every=10, resolution=(64, 64)):
        super().__init__()
        self.model_ref = model_ref  # NeuralImageGenerator instance
        self.save_every = save_every
        self.resolution = resolution
        self.frames = []
        self.image_index = 0

    
    def on_epoch_end(self, epoch, logs=None):
        if epoch % self.save_every == 0:
            print(Fore.YELLOW + f"\n[VideoCallback] Generating frame at epoch {epoch}..." + Style.RESET_ALL)
            coords = self.model_ref.images_inputs[self.image_index]  # Already full-size input
            preds = self.model_ref.model(coords, training=False).numpy()
            frame = (preds.reshape((*self.resolution[::-1], 3)) * 255).astype(np.uint8)
            self.frames.append(frame)


    def generate_frame(self):
        preds, size = self.model_ref.get_prediction(self.image_index, hyper_res=False)
        
        # Reshape into image
        frame = (preds.reshape((*self.resolution[::-1], 3)) * 255).astype(np.uint8)
        self.frames.append(frame)


    def save_video(self, fps, output_path):
        if not self.frames:
            print(Fore.RED + "[VideoCallback] No frames to save." + Style.RESET_ALL)
            return

        # Ensure output directory exists
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            try:
                os.makedirs(output_dir, exist_ok=True)
                print(Fore.YELLOW + f"[VideoCallback] Created directory {output_dir}" + Style.RESET_ALL)
            except Exception as e:
                print(Fore.RED + f"[VideoCallback] Failed to create directory {output_dir}: {e}" + Style.RESET_ALL)
                return

        height, width = self.resolution[1], self.resolution[0]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        if not out.isOpened():
            print(Fore.RED + "[VideoCallback] Failed to open VideoWriter. Check file path and codec." + Style.RESET_ALL)
            return

        for i, frame in enumerate(self.frames):
            if frame.shape[1] != width or frame.shape[0] != height:
                print(Fore.RED + f"[VideoCallback] Skipping frame {i}: wrong resolution ({frame.shape[1]}x{frame.shape[0]}), expected {width}x{height}." + Style.RESET_ALL)
                continue
            try:
                out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            except Exception as e:
                print(Fore.RED + f"[VideoCallback] Error writing frame {i}: {e}" + Style.RESET_ALL)

        out.release()

        if os.path.isfile(output_path):
            print(Fore.GREEN + f"[VideoCallback] Video successfully saved to {output_path}" + Style.RESET_ALL)
        else:
            print(Fore.RED + f"[VideoCallback] Video file not found after saving attempt: {output_path}" + Style.RESET_ALL)

    
    
    def reset(self, resolution, image_index):
        self.resolution = resolution
        self.image_index = image_index
        self.frames = []
