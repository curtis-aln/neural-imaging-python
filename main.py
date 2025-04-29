# TODO:
# Push code to github
# allow file saving for new model
# autosaving every n minutes
# ability to generate a video at the end

from neural_image import *
from matplotlib_rendering import MatplotLibRendering

img_generator = NeuralImageGenerator(load_model=True)
prediction, evalutation, size = img_generator.train_model(save_model=True)

print(Fore.GREEN + '\nTest loss:' + Style.RESET_ALL, round(evalutation))

rendering = MatplotLibRendering(img_generator)

rendering.render(prediction, size)

