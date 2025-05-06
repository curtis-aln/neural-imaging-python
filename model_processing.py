import tensorflow as tf

from network.net_utils import *
from settings import *

model_name = 'model' # todo
training_video_name = "yingyang"

shortest_side_length = 200 #int(input("what image shortest length would you like? "))
frames_per_second = 30 #int(input("fps? "))
video_length_seconds = 5 #int(input("video length? "))

def main():
    model = tf.keras.models.load_model(model_save_folder_path) # todo error detection
    print(f"model has been loaded from '{model_save_folder_path}'")
    
    frames = frames_per_second * video_length_seconds
    print(f"{frames} frames will be generated")

    path = video_dataset_path + training_video_name + ".mp4"
    original_video, size = load_video_from_file(path, shortest_side_length, frames)
    print(f"Training Video loaded from path '{path}'")
    print(f"Size of ({size[0]}, {size[1]})")
    print(f"Total pixels per frame: {size[0] * size[1]}")
    print(f"Total pixels overall: {size[0] * size[1] * frames}")
    
    input_data = create_video_input_data(size, frames)
    print(f"input data created with shape {input_data.shape}")

    prediction = model.predict(input_data)
    print("Predictions have been created, now formatting into .mp4")
    save_path = final_predictions_save_path + "hyper_res.mp4"
    save_flat_predictions_as_video(prediction, save_path, original_video.shape, frames_per_second)
    print(f"hyper-resolution video has been saved to '{save_path}'")


    


if __name__ == '__main__':
    main()