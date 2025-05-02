from network.siren_model import ModelConfig

training_image_folder = "media/training_images/"
video_save_path = "media/timelapse_videos/"
final_image_save_path = "media/final_images/"

weights_save_path = "outputs/network_data.weights.h5" # todo

epochs_per_image = 1500
video_frame_rate = 30

# the longest length of an image, for lanscape images its width will be this, height for portrait images
image_longest_length = 256


 # creating the model we use for training
config = ModelConfig(
    input_dim=7,
    hidden_layers=15,
    hidden_units=180,
    w0=1.0,
    w0_initial=30.0,
    final_activation='sigmoid'
)

