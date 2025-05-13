from network.neural_image import *
from matplotlib_rendering import PredictionSlideshow

# when continuing to train a video the mp4 file needs to be overwritten
# smart overwriting for video training, add a "session2" to the end or something
# A Load models ui which tells you the shape and training times of all the current saved models
# - as well as the most recently accessed model
# A timer is needed for how long the training will take
# trained image final result should be saved too
# logs should tell us information about the video saved - file size, frame rate, video length, bitrate, etc

img_generator = NeuralImageGenerator()
predictions, image_sizes, training_images = img_generator.train_model()

rendering = PredictionSlideshow(predictions, image_sizes, training_images)

