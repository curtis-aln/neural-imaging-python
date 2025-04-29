# TODO:
# Ability to continue adding to the same video
# Hyper-resolution video generation mode
# multiple save files
# program can train through a whole file of photos
# program can train on videos

# Inquireries
# What happens if we feed it two images training data
# seemless transitions

from neural_image import *
from matplotlib_rendering import MatplotLibRendering

img_generator = NeuralImageGenerator(load_model=True)
prediction, evalutation, size = img_generator.train_model(save_model=True)

print(Fore.GREEN + '\nTest loss:' + Style.RESET_ALL, round(evalutation))

rendering = MatplotLibRendering(img_generator)

rendering.render(prediction, size)

