from network.siren_model import ModelConfig

image_dataset_path = "media/training_images/"
video_dataset_path = "media/video_dataset/"
timelapse_save_path = "media/timelapse_videos/"
final_predictions_save_path = "media/final_images/"

model_save_folder_path = "model_saves/model.keras" # todo name should change for each different training session

# how many training generations each image will recive before moving onto the next #todo improve
epochs_per_image = 200

# when generating the timelapse of the training, this will be its fps
timelapse_fps = 40

# the longest length of an image, for lanscape images its width will be this, height for portrait images
image_longest_length = 100

""" Video training """
frames_max = 40
video_predictions_fps = 10
video_batch_size = 20_000 # image batch size is by defualt the maximum #todo

video_generation = False
load_model = False

 # creating the model we use for training
config = ModelConfig(
    input_dim=8,
    hidden_layers=10,
    hidden_units=180,
    w0=1.0,
    w0_initial=30.0,
    final_activation='sigmoid'
)

model_optimizer = 'adam'
model_loss = 'mse'

