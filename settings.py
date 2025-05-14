from network.siren_model import ModelConfig

image_dataset_path = "media/image_dataset/"
video_dataset_path = "media/video_dataset/"
timelapse_save_path = "media/timelapse_videos/"
final_predictions_save_path = "media/final_images/"

model_save_folder_path = "model_saves/" # todo name should change for each different training session

# if you only want to train one image / video then specify its name here | make sure its extension (.png / .mp4) is included
specific_media_to_train = "loft.png"

# how many training generations each image will recive before moving onto the next #todo improve
epochs_per_image = 2000

forier_count = 6 #todo

# when generating the timelapse of the training, this will be its fps
timelapse_fps = 50

# the longest length of an image, for lanscape images its width will be this, height for portrait images
image_longest_length = 400

""" Video training """
frames_max = 20
video_predictions_fps = 10
batch_size = 8000 # image batch size is by defualt the maximum #todo

 # creating the model we use for training
config = ModelConfig(
    input_dim=30, # todo
    hidden_layers=7,
    hidden_units=125,
    w0=1.0,
    w0_initial=30.0,
    final_activation='sigmoid'
)

model_optimizer = 'adam'
model_loss = 'mse'

