import tensorflow as tf
import numpy as np
from tensorflow import keras

from network.net_utils import *
from settings import *

def main():
    #name = input("What is the name of the model you would like to load? ")
    #todo

    model = tf.keras.models.load_model(model_save_folder_path)
    print("model has been loaded")
    

    shortest_len = int(input("what image shortest length would you like? "))
    fps = int(input("fps? "))
    time = int(input("video length? "))
    frames = fps * time

    video_correspondent = input("What video was this model trained on? ") # todo images should work too
    path = video_dataset_path + video_correspondent + ".mp4"
    original_video = load_video_from_file(path, shortest_len, frames)
    size = (original_video[1], original_video[2])
    print("Video loaded and has a size of " + size)
    
    input_data = create_video_input_data(size, frames)
    print("input data created")

    prediction = model.predict(input_data)
    save_flat_predictions_as_video(prediction, final_predictions_save_path + "hyper_res.mp4", original_video.shape, fps)
    print("saved")


    


if __name__ == '__main__':
    main()