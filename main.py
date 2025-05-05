# TODO:
# Video weights saving with cusotom name
# Ability to continue adding to the same video
# Two save files - original resolution and hyper resolution (with boosted fps)

# Hyper-resolution video generation mode
# program can train on videos

from network.neural_image import *
from matplotlib_rendering import PredictionSlideshow

img_generator = NeuralImageGenerator(load_model=load_model)
predictions, image_sizes, training_images = img_generator.train_model()

rendering = PredictionSlideshow(predictions, image_sizes, training_images)

