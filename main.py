# TODO:
# Ability to continue adding to the same video
# Hyper-resolution video generation mode
# multiple save files
# program can train through a whole file of photos
# program can train on videos

# Inquireries
# What happens if we feed it two images training data
# seemless transitions

# bee took about 5000 epochs

from network.neural_image import *
from matplotlib_rendering import MatplotLibRendering

img_generator = NeuralImageGenerator(load_model=True)
predictions, image_sizes = img_generator.train_model(save_model=True)


rendering = MatplotLibRendering(img_generator)

#rendering.render(prediction, size)

