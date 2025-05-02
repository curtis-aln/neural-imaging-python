# TODO:
# Ability to continue adding to the same video
# Hyper-resolution video generation mode
# program can train on videos

from network.neural_image import *
from matplotlib_rendering import PredictionSlideshow

img_generator = NeuralImageGenerator(load_model=True)
predictions, image_sizes, training_images = img_generator.train_model(save_model=True)

rendering = PredictionSlideshow(predictions, image_sizes, training_images)

