from settings import *
from network.net_utils import *

frames = frames_max
image, size = load_video_from_file("media/video_dataset/yingyang.mp4", image_longest_length, frames)
print(f"image shape: {image.shape} with size {size}")
shape = image.shape

flattened = normalize_and_reshape_media(image, False)
print(f"flattened shape: {flattened.shape}")

save_flat_predictions_as_video(flattened, "media/final_images/proof.mp4", shape, timelapse_fps)