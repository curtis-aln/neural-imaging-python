from network.siren_model import ModelConfig

image_dataset_path = "media/training_images/"
video_dataset_path = "media/video_dataset/"
timelapse_save_path = "media/timelapse_videos/"
final_predictions_save_path = "media/final_images/"

weights_save_path = "outputs/network_data.weights.h5" # todo

epochs_per_image = 2000
video_frame_rate = 10
frames_max = 40

# the longest length of an image, for lanscape images its width will be this, height for portrait images
image_longest_length = 40

video_predictions_fps = 10
video_batch_size = 60_000

video_generation = False
load_model = False

 # creating the model we use for training
config = ModelConfig(
    input_dim=8,
    hidden_layers=6,
    hidden_units=100,
    w0=1.0,
    w0_initial=30.0,
    final_activation='sigmoid'
)

model_optimizer = 'adam'
model_loss = 'mse'

