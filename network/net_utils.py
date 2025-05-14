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
# todo orginize
from progress_bar import ProgressBar

import gc


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
def load_image_from_file(image_path: str, desired_shortest_side: int) -> tuple[np.ndarray, tuple]:
    img = cv2.imread(image_path)
    if img is None:
        text = f"Image at {image_path} could not be read."
        raise ValueError(text)
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    h, w = img.shape[:2]
    if h < w:
        scale = desired_shortest_side / h
    else:
        scale = desired_shortest_side / w

    new_size = (int(w * scale), int(h * scale))
    return cv2.resize(img, new_size, interpolation=cv2.INTER_CUBIC), new_size

# Function to load a single videoimport cv2
import numpy as np

def load_video_from_file(video_path: str, desired_shortest_side: int, frame_count: int) -> tuple[np.ndarray, tuple]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Video at {video_path} could not be read.")
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    selected_indices = np.linspace(0, total_frames - 1, frame_count).astype(int)
    
    frames = []
    new_size = (0, 0)
    current_index = 0
    selected_pos = 0  # index in selected_indices

    while selected_pos < len(selected_indices) and current_index < total_frames:
        if current_index == selected_indices[selected_pos]:
            cap.set(cv2.CAP_PROP_POS_FRAMES, current_index)
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w = frame.shape[:2]

            # Scale preserving aspect ratio
            scale = desired_shortest_side / min(h, w)
            new_size = (int(w * scale), int(h * scale))
            resized_frame = cv2.resize(frame, new_size, interpolation=cv2.INTER_CUBIC)

            frames.append(resized_frame)
            selected_pos += 1

        current_index += 1

    cap.release()
    return np.array(frames), new_size



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
                img, _ = load_image_from_file(media_path, desired_shortest_side)
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



def add_fourier_features(values: np.ndarray, freqs: list[float]) -> np.ndarray:
    """Applies Fourier feature mapping to the given values."""
    features = []
    for freq in freqs:
        features.append(np.sin(2 * np.pi * freq * values))
        features.append(np.cos(2 * np.pi * freq * values))
    return np.concatenate(features, axis=1)

def generate_fourier_frequencies(n: int) -> list[int]:
    return [2 ** i for i in range(n)]


def create_input_data(image_size: tuple, fourier_freqs_xy=[1, 2, 4, 8]) -> np.ndarray:
    size_x, size_y = image_size

    # Normalized coordinates centered at (0,0), range [-1, 1]
    x, y = np.meshgrid(
        np.linspace(-1, 1, size_x),
        np.linspace(-1, 1, size_y),
        indexing='xy'
    )
    x_flat = x.ravel().reshape(-1, 1)
    y_flat = y.ravel().reshape(-1, 1)

    # Polar coordinates
    r = np.sqrt(x_flat**2 + y_flat**2)
    theta = np.arctan2(y_flat, x_flat)
    polar_space = np.concatenate([r, np.sin(theta), np.cos(theta)], axis=1)

    # Fourier-encoded spatial features
    xy = np.concatenate([x_flat, y_flat], axis=1)
    xy_fourier = add_fourier_features(xy, fourier_freqs_xy)

    # Placeholder for time (to be filled in later)
    time_column = np.zeros((x_flat.shape[0], 1))

    return np.concatenate([xy, polar_space, xy_fourier, time_column], axis=1)


def create_video_input_data(
    image_size: tuple,
    frames: int,
    max_time_value=1.0,
    fourier_freqs_xy=[1, 2, 4, 8],
    fourier_freqs_t=[1, 2, 4]
) -> np.ndarray:
    base_input = create_input_data(image_size, fourier_freqs_xy)
    num_pixels = image_size[0] * image_size[1]
    time_column_index = base_input.shape[1] - 1

    video_input = np.repeat(base_input[None, :, :], frames, axis=0)

    # Add time values and Fourier-encoded time features
    time_features = []
    for i in range(frames):
        t = (i / (frames - 1)) * max_time_value if frames > 1 else 0.0
        video_input[i, :, time_column_index] = t
        time_features.append(add_fourier_features(np.full((num_pixels, 1), t), fourier_freqs_t))

    time_features_stacked = np.stack(time_features, axis=0)  # shape: (frames, num_pixels, time_features)
    
    # Final concatenation
    video_input = np.concatenate([video_input, time_features_stacked], axis=2)

    return video_input.reshape(-1, video_input.shape[2])  # (frames * pixels, features)




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
    """
    Reshape flattened normalized data into media (uint8).
    Supports both full video and single-frame input.
    
    - flat_data: (frames * height * width, 3) or (height * width, 3)
    - original_shape: (frames, height, width, 3) or (1, height, width, 3)
    """
    dims = len(original_shape)
    if dims != 4:
        raise ValueError("original_shape must be 4D (frames, height, width, 3)")

    frames, height, width, _ = original_shape
    pixels_per_frame = height * width

    if flat_data.shape[0] == pixels_per_frame:
        # Single frame
        reshaped = flat_data.reshape(height, width, 3)
    elif flat_data.shape[0] == frames * pixels_per_frame:
        # Full video
        reshaped = flat_data.reshape(frames, height, width, 3)
    else:
        raise ValueError(f"Input shape mismatch: expected {pixels_per_frame} or {frames * pixels_per_frame} pixels, got {flat_data.shape[0]}")

    return (reshaped * 255).astype(np.uint8)



def save_flat_predictions_as_video(flat_predictions, output_path, original_shape, frame_rate=30, extra_info=True):
    num_frames, height, width, _ = original_shape

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, frame_rate, (width, height))

    if extra_info:
        progress = ProgressBar(total=num_frames, prefix='Saving video', suffix='frames')

    pixels_per_frame = height * width
    for i in range(num_frames):
        start = i * pixels_per_frame
        end = (i + 1) * pixels_per_frame

        # Shape: (pixels, 3)
        frame_flat = flat_predictions[start:end]
        
        # Reshape to (height, width, 3)
        frame = frame_flat.reshape((height, width, 3))
        
        # Convert from float32 (0..1) to uint8 (0..255) if needed
        if frame.dtype != np.uint8:
            frame = np.clip(frame * 255, 0, 255).astype(np.uint8)

        out.write(frame)

        if extra_info:
            progress.update(i + 1)

        # Optional: release memory if needed
        del frame, frame_flat
        gc.collect()

    out.release()
    print(Fore.MAGENTA + f"Video saved to {output_path}" + Style.RESET_ALL)

