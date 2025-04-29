import numpy as np
import cv2
import os

# Create frames directory
os.makedirs("frames", exist_ok=True)

num_frames = 300
for i in range(num_frames):
    frame = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
    filename = f"frames/frame_{i:04d}.png"
    cv2.imwrite(filename, frame)

print("Frames saved. Now run FFmpeg to make the video.")
# ffmpeg -framerate 30 -i frames/frame_%04d.png -c:v libx264 -crf 18 -pix_fmt yuv420p random_video.mp4