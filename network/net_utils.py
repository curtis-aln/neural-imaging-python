# Neural Networking
import tensorflow as tf
import numpy as np

# video recording & file saving
import cv2
import os

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


class LossHistory(tf.keras.callbacks.Callback):
    def __init__(self):
        super().__init__()
        self.losses = []

    def on_epoch_end(self, epoch, logs=None):
        if logs is not None and 'loss' in logs:
            self.losses.append(logs['loss'])


import os
import cv2
import numpy as np

# Function to load a single image
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

# Function to load a single video
def load_video_from_file(video_path: str, desired_shortest_side: int) -> np.ndarray:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Video at {video_path} could not be read.")
    
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert each frame to RGB and resize
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = frame.shape[:2]
        if h < w:
            scale = desired_shortest_side / h
        else:
            scale = desired_shortest_side / w
        
        new_size = (int(w * scale), int(h * scale))
        resized_frame = cv2.resize(frame, new_size, interpolation=cv2.INTER_CUBIC)
        frames.append(resized_frame)
    
    cap.release()
    return np.array(frames)  # Return a numpy array of frames

# General function to load either images or videos from a folder
def load_all_media_from_folder(folder_path: str, desired_shortest_side: int, media_type: str = 'images') -> list:
    supported_image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp'}
    supported_video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.flv'}

    # todo exeption for mediatype

    media_files = []
    file_names = os.listdir(folder_path)

    for filename in file_names:
        file_extension = os.path.splitext(filename)[1].lower()
        media_path = os.path.join(folder_path, filename)
        
        if media_type == 'images' and file_extension in supported_image_extensions:
            try:
                img = load_image_from_file(media_path, desired_shortest_side)
                media_files.append(img)
            except Exception as e:
                print(f"Skipping {filename}: {e}")
        
        elif media_type == 'videos' and file_extension in supported_video_extensions:
            try:
                video_frames = load_video_from_file(media_path, desired_shortest_side)
                media_files.append(video_frames)
            except Exception as e:
                print(f"Skipping {filename}: {e}")

    file_names_no_ext = [os.path.splitext(name)[0] for name in file_names]  # Remove file extension
    return media_files, file_names_no_ext



def get_window_dims():
    root = tk.Tk()
    width = root.winfo_screenwidth()
    height = root.winfo_screenheight()
    root.destroy()
    return width, height



def create_input_data(size):
    """
    Creates input data for images or videos. The input data includes spatial
    coordinates (Cartesian and polar) and an additional time value for videos.
    The time value is set to 0 for images.
    
    :param size: Tuple of size (sx, sy) for images or (sx, sy, num_frames) for videos
    :return: Concatenated array of spatial and time-related data
    """
    # Check if size has 3 dimensions (indicating video with time)
    if len(size) == 3:
        sx, sy, num_frames = size
        time_vals = np.linspace(0, 1, num_frames)  # Create a time range from 0 to 1
    else:
        sx, sy = size
        time_vals = np.zeros(1)  # For images, time is always 0
    
    x, y = np.meshgrid(np.linspace(0, 1, sx), np.linspace(0, 1, sy), indexing='xy')

    # Cartesian Top-Left & Bottom-Right
    input_space_TL = np.column_stack((x.ravel(), y.ravel()))
    input_space_BR = np.column_stack(((1 - x).ravel(), (1 - y).ravel()))

    # Polar (centered)
    dx, dy = x - 0.5, y - 0.5
    r = np.sqrt(dx**2 + dy**2)
    theta = np.arctan2(dy, dx)
    polar_space = np.column_stack((r.ravel(), np.sin(theta).ravel(), np.cos(theta).ravel()))

    # Add the time values (broadcasting time_val across all spatial points)
    time_space = np.repeat(time_vals, sx * sy).reshape(-1, 1)
    
    # [TL.x, TL.y, BR.x, BR.y, R, Sin, Cos, time_val]
    return np.concatenate((input_space_TL, input_space_BR, polar_space, time_space), axis=1)


def normalize_and_reshape_media(media, media_type='image'):
    """
    Normalizes and reshapes the media (either an image or a video).
    
    :param media: The media to process (image or video)
    :param media_type: 'image' or 'video', to specify the type of media
    :return: A reshaped and normalized array
    """
    # Normalize media to range [0, 1]
    normalized = media / 255.0

    if media_type == 'image':
        height, width = media.shape[:2]
        return normalized.reshape(height * width, 3)

    elif media_type == 'video':
        return normalized.reshape(-1)  # Flatten all frames into 1D



def convert_predictions_to_video(predictions, output_path, frame_rate=30, resolution=(256, 256)):
    """
    Converts a sequence of network predictions (frames) into an MP4 video.

    :param predictions: List or array of frames (each frame should be of shape (height, width, 3)).
    :param output_path: Path where the output video will be saved (e.g., 'output_video.mp4').
    :param frame_rate: Frame rate of the video (default is 30 fps).
    :param resolution: Resolution of the video (height, width), defaults to (256, 256).
    :return: None
    """
    # Initialize the VideoWriter object with the desired output path, codec, frame rate, and resolution
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 'mp4v' is the codec for MP4 files
    out = cv2.VideoWriter(output_path, fourcc, frame_rate, resolution)

    # Write each frame to the video
    for frame in predictions:
        # Ensure the frame is in the correct format (uint8, RGB)
        frame = np.clip(frame, 0, 255).astype(np.uint8)
        
        # Resize frame if necessary
        if frame.shape[0] != resolution[0] or frame.shape[1] != resolution[1]:
            frame = cv2.resize(frame, resolution)

        # Write the frame to the video
        out.write(frame)

    # Release the VideoWriter object after writing all frames
    out.release()
    print(f"Video saved to {output_path}")
