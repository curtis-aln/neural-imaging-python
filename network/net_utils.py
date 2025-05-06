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
def load_video_from_file(video_path: str, desired_shortest_side: int, frame_count) -> np.ndarray:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Video at {video_path} could not be read.")
    
    frames = []
    new_size = (0, 0)

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
    reduced = reduce_video_to_frame_count(np.array(frames), frame_count)
    return reduced, new_size  # Return a numpy array of frames


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
                video_frames, _ = load_video_from_file(media_path, desired_shortest_side, frame_count)
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


def create_input_data(image_size : tuple) -> np.ndarray:
    size_x, size_y = image_size
    
    # cartesian coordinates
    x, y = np.meshgrid(np.linspace(0, 1, size_x), np.linspace(0, 1, size_y), indexing='xy')
    
    input_space_TL = np.column_stack((x.ravel(), y.ravel()))
    input_space_BR = np.column_stack(((1 - x).ravel(), (1 - y).ravel()))
    
    # polar coordinates
    dx, dy = x - 0.5, y - 0.5
    r = np.sqrt(dx**2 + dy**2)
    theta = np.arctan2(dy, dx)
    polar_space = np.column_stack((r.ravel(), np.sin(theta).ravel(), np.cos(theta).ravel()))
    time_space = np.zeros((size_x * size_y, 1))

    # adding them all together
    return np.concatenate((input_space_TL, input_space_BR, polar_space, time_space), axis=1)

def create_video_input_data(image_size: tuple, frames: int) -> np.ndarray:
    """Creates input data for a video where each frame has time embedded in the input features."""
    base_input = create_input_data(image_size)
    num_pixels = image_size[0] * image_size[1]
    
    # Index of the time column (last column in base_input)
    time_column_index = base_input.shape[1] - 1
    
    video_input = np.repeat(base_input[None, :, :], frames, axis=0)  # shape: (frames, num_pixels, features)

    # Add time encoding to each frame
    for i in range(frames):
        time_value = i / (frames - 1) if frames > 1 else 0.0
        video_input[i, :, time_column_index] = time_value

    return video_input.reshape(-1, base_input.shape[1])  # Flatten to (frames * num_pixels, features)



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

def reshape_normalized_back_to_media(flat_data, original_shape):
    frames, height, width, _ = original_shape
    reshaped = flat_data.reshape(frames, height, width, 3)
    return (reshaped * 255).astype(np.uint8)

def save_flat_predictions_as_video(flat_predictions, output_path, original_shape, frame_rate=30):
    # Reconstruct original video tensor
    reconstructed = reshape_normalized_back_to_media(flat_predictions, original_shape)
    print("Video data reshaped")
    
    num_frames, height, width, _ = reconstructed.shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, frame_rate, (width, height))

    print(reconstructed.shape)
    for i in range(num_frames):
        frame = reconstructed[i]
        out.write(frame)
    out.release()

    print(Fore.MAGENTA + f"video saved to {output_path}" + Style.RESET_ALL) 