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


def reduce_video_to_frame_count(video_array: np.ndarray, frame_count: int) -> np.ndarray:
    """
    Reduces the number of frames in a video to the desired count by uniform skipping.

    :param video_array: A NumPy array of shape (num_frames, height, width, 3)
    :param frame_count: The desired number of frames
    :return: A NumPy array of shape (frame_count, height, width, 3)
    """
    original_frame_count = video_array.shape[0]

    if frame_count >= original_frame_count:
        print(f"Requested {frame_count} frames, but video only has {original_frame_count}. Returning original.")
        return video_array

    # Get indices to sample uniformly
    indices = np.linspace(0, original_frame_count - 1, frame_count, dtype=int)
    reduced_video = video_array[indices]

    return reduced_video


# General function to load either images or videos from a folder
def load_all_media_from_folder(folder_path: str, desired_shortest_side: int, media_type: str = 'images', frame_count = 60) -> list:
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
                reduced = reduce_video_to_frame_count(video_frames, frame_count)
                media_files.append(reduced)

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

    :param size: Tuple of size (sx, sy) for images or (num_frames, sx, sy) for videos
    :return: Concatenated array of spatial and time-related data
    """
    if len(size) == 4: # frame, x, y, col
        num_frames, sx, sy, col = size
        time_vals = np.linspace(0, 1, num_frames)

        # Spatial grid
        x, y = np.meshgrid(np.linspace(0, 1, sx), np.linspace(0, 1, sy), indexing='xy')
        input_space_TL = np.column_stack((x.ravel(), y.ravel()))
        input_space_BR = np.column_stack(((1 - x).ravel(), (1 - y).ravel()))
        dx, dy = x - 0.5, y - 0.5
        r = np.sqrt(dx**2 + dy**2)
        theta = np.arctan2(dy, dx)
        polar_space = np.column_stack((r.ravel(), np.sin(theta).ravel(), np.cos(theta).ravel()))

        # Repeat spatial info for each frame
        spatial = np.concatenate((input_space_TL, input_space_BR, polar_space), axis=1)
        spatial_repeated = np.tile(spatial, (num_frames, 1))

        # Time values for each frame
        time_space = np.repeat(time_vals, sx * sy).reshape(-1, 1)

        return np.concatenate((spatial_repeated, time_space), axis=1)


    else:
        sx, sy = size
        x, y = np.meshgrid(np.linspace(0, 1, sx), np.linspace(0, 1, sy), indexing='xy')
        input_space_TL = np.column_stack((x.ravel(), y.ravel()))
        input_space_BR = np.column_stack(((1 - x).ravel(), (1 - y).ravel()))
        dx, dy = x - 0.5, y - 0.5
        r = np.sqrt(dx**2 + dy**2)
        theta = np.arctan2(dy, dx)
        polar_space = np.column_stack((r.ravel(), np.sin(theta).ravel(), np.cos(theta).ravel()))
        time_space = np.zeros((sx * sy, 1))

        return np.concatenate((input_space_TL, input_space_BR, polar_space, time_space), axis=1)



def normalize_and_reshape_media(media, is_image):
    """
    Normalizes and reshapes the media (either an image or a video).
    
    :param media: The media to process (image or video)
    :param media_type: 'image' or 'video', to specify the type of media
    :return: A reshaped and normalized array
    """
    # Normalize media to range [0, 1]
    normalized = media / 255.0

    if is_image:
        height, width = media.shape[:2]
        return normalized.reshape(height * width, 3)

    shape = media.shape
    flattened = normalized.reshape(shape[0] * shape[1] * shape[2], shape[3])
    return flattened


def convert_predictions_to_video(predictions, output_path, num_frames, resolution, frame_rate=30):
    """
    Converts flattened network predictions into an MP4 video.

    :param predictions: Flattened array of shape (num_frames * height * width, 3).
    :param output_path: Path to save the output video (e.g. 'output.mp4').
    :param num_frames: Number of frames in the video.
    :param resolution: (height, width) of each frame.
    :param frame_rate: FPS of the output video.
    """
    height, width = resolution
    pixels_per_frame = height * width

    # Check for consistent size
    expected_total_pixels = num_frames * pixels_per_frame
    if predictions.shape[0] != expected_total_pixels:
        raise ValueError(f"Prediction size mismatch: expected {expected_total_pixels} pixels, got {predictions.shape[0]}")

    # Reshape predictions to (num_frames, height, width, 3)
    reshaped = predictions.reshape((num_frames, height, width, 3))

    # Initialize VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, frame_rate, (width, height))

    for frame in reshaped:
        # Convert from RGB to BGR for OpenCV
        frame_bgr = cv2.cvtColor(np.clip(frame, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        out.write(frame_bgr)

    out.release()
    print(f"Video saved to {output_path}")
