import tensorflow as tf
import numpy as np
import psutil
import time

from network.net_utils import *
from settings import *

# ========== SETTINGS ==========
model_name = 'yingyang_video'
training_video_name = "yingyang"

shortest_side_length = 380
frames_per_second = 15
video_length_seconds = 4

batch_size = 8192
use_mixed_precision = True
# ==============================

def enable_mixed_precision():
    try:
        tf.keras.mixed_precision.set_global_policy("mixed_float16")
        print("[INFO] Mixed precision enabled (float16)")
    except:
        print("[WARNING] Mixed precision not supported on this system.")

def print_system_resources():
    vm = psutil.virtual_memory()
    print(f"[SYSTEM] RAM used: {vm.used / 1024 ** 3:.2f} GB / {vm.total / 1024 ** 3:.2f} GB")
    print(f"[SYSTEM] Available: {vm.available / 1024 ** 3:.2f} GB")

def create_batched_dataset(input_data, batch_size):
    dataset = tf.data.Dataset.from_tensor_slices(input_data)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset

def main():
    if use_mixed_precision:
        enable_mixed_precision()

    print_system_resources()

    # === Load model ===
    path = model_save_folder_path + model_name + ".keras"
    model = tf.keras.models.load_model(path)
    print(f"[INFO] Model loaded from '{path}'")

    # === Load video ===
    frames = frames_per_second * video_length_seconds
    video_path = video_dataset_path + training_video_name + ".mp4"
    original_video, size = load_video_from_file(video_path, shortest_side_length, frames)

    print(f"[INFO] Video loaded: {video_path}")
    print(f"[INFO] Video size: {size}, Frames: {frames}")
    print(f"[INFO] Total pixels overall: {size[0] * size[1] * frames}")

    # === Prepare input data ===
    input_data = create_video_input_data(size, frames, max_time_value=2)
    print(f"[INFO] Input data created: shape {input_data.shape}, dtype {input_data.dtype}")

    dataset = create_batched_dataset(input_data, batch_size)

    # === Predict in batches ===
    print("[INFO] Running inference...")
    start_time = time.time()
    predictions = model.predict(dataset, verbose=1)
    print(f"[INFO] Inference done in {time.time() - start_time:.2f} seconds")

    # === Save output video ===
    output_path = final_predictions_save_path + "hyper_res.mp4"
    save_flat_predictions_as_video(predictions, output_path, original_video.shape, frames_per_second)
    print(f"[INFO] Saved hyper-resolution video to '{output_path}'")


if __name__ == '__main__':
    main()
