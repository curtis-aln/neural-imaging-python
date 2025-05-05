# Neural Networking
import tensorflow as tf
import numpy as np
from network.siren_model import build_siren_model

# video recording & file saving
import os

from network.video_callback import VideoCallback

# nice printing colors
from colorama import Fore, Style

from settings import *

from network.net_utils import *

# we have both videos and images possible so we use this function to differentiate
def get_media_shapes(data, training_images = True):
    if training_images:
        return [(img.shape[1], img.shape[0]) for img in data]
    
    return [(video.shape[2], video.shape[1]) for video in data]

class NeuralImageGenerator:
    def __init__(self, load_model = False):
        # loading the images and the videos from memory
        images, image_names = load_all_media_from_folder(image_dataset_path, image_longest_length, media_type='images')
        videos, video_names = load_all_media_from_folder(video_dataset_path, image_longest_length, media_type='videos', frame_count=frames_max)

        # determining whether the media type is video or image
        self.is_training_images = False #len(images) != 0

        img_len, vid_len = len(images), len(videos)
        #if (img_len != 0 and vid_len != 0):
        #    print(Fore.MAGENTA + "There are both images and videos detected in the training folders, would you like videos (v) to be trained or images (i)? ")
         #   self.is_training_images = input(Fore.MAGENTA + ">>> " + Style.RESET_ALL) == "i"
        
        self.training_media, self.training_names = (images, image_names) if self.is_training_images else (videos, video_names)
        
        # each media will have their own size and aspect ratio which needs to be fetched to create their own input space
        self.media_frame_sizes = get_media_shapes(self.training_media, self.is_training_images)


        # Flattening the data so that we can map it to the input space and feed into into the tensorflow trainer
        self.normalized_media = [normalize_and_reshape_media(media, self.is_training_images) for media in self.training_media]

        # creating the input data from the media sizes | automatically adjusts for videos
        self.input_space = [create_input_data(video.shape) for video in videos]

        # creating the training datasets
        print("videos shape", videos[0].shape)
        print("inp space shape", self.input_space[0].shape)
        print("normalized media shape", self.normalized_media[0].shape)
        self.datasets = [self.create_dataset(inp, norm) for inp, norm in zip(self.input_space, self.normalized_media)]

        # loading and compiling the model using SIREN
        self.model = build_siren_model(config)
        if load_model:
            self.load_model()
        self.model.summary()
        
        self.model.compile(optimizer=model_optimizer, loss=model_loss)

        # at the end of training each model we store its best prediction of the training data here
        self.preditions = []
        
        # Important callbacks during training
        self.video_callback = VideoCallback(self, save_every=1, resolution=self.media_frame_sizes[0])
        cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=weights_save_path, 
                                                         save_weights_only=True, verbose=1, mode='max', save_best_only=True)
        self.callbacks = [LossHistory(), cp_callback, SingleLineLogger()]

        if self.is_training_images:
            self.callbacks.append(self.video_callback)


    def train_model(self, hyper_res=False) -> tuple: # todo give these params a use        
        
        image_index = 0 # each model is trained one after the other
        for dataset, name, size in zip(self.datasets, self.training_names, self.media_frame_sizes):
            self.video_callback.reset(size, image_index)

            exeption = self.fit_image(dataset, self.callbacks, name)
            prediction = self.get_prediction(image_index, hyper_res)[0]
            
            if self.is_training_images:
                print("==== Saving Video ==== ")
                path = timelapse_save_path + name + ".mp4"
                print(Fore.BLUE + f"saving to path '{path}'" + Style.RESET_ALL)
                self.video_callback.save_video(fps=video_frame_rate, output_path=path)

            else:
                path = final_predictions_save_path + name + ".mp4"
                shape = self.training_media[image_index].shape
                save_flat_predictions_as_video(prediction, path, shape, video_predictions_fps)
            
            print("==== Evaluating model ==== ")
            self.preditions.append(prediction)
            
            if exeption:
                break
            image_index += 1
        
        quit()
        return self.preditions, self.media_frame_sizes, self.training_media


    def fit_image(self, dataset, call_backs, name):
        # training this image
        exeption = False # if the user interupts we want to stop the program
        try: 
            self.model.fit(
                    dataset, 
                    epochs=epochs_per_image,
                    callbacks=call_backs,
                    verbose=0)
        except KeyboardInterrupt: 
            print(Fore.RED + "\nTraining stopped. Compiling Results. . ." + Style.RESET_ALL)
            exeption = True

        print(Fore.MAGENTA + f"Training finished for {name}" + Style.RESET_ALL)
        return exeption
            
        
   
    def get_prediction(self, index, hyper_res=False) -> tuple[np.ndarray, tuple]:
        original_size = self.media_frame_sizes[index]
        inp_data = self.input_space[index]

        if not hyper_res:
            return self.model.predict(inp_data), original_size

        size = original_size * 4
        input_space = self.create_input_data(size)
        return self.model.predict(input_space), size

    def get_losses(self):
        return self.loss_callback.losses


    def create_dataset(self, inp_data, normalised_img):
        dataset = tf.data.Dataset.from_tensor_slices((inp_data, normalised_img))

        batch_size = inp_data.shape[0] if self.is_training_images else video_batch_size
        dataset = dataset.batch(batch_size)
        return dataset


    def load_model(self):
        if not os.path.exists(weights_save_path):
            print(Fore.RED + "No previous save file exists, creating new model" + Style.RESET_ALL)
            return
        
        print(Fore.BLUE + f"A model has been retrived from '{weights_save_path}'")
        if input(Fore.BLUE + "would you like to continue training with this model? y/n " + Style.RESET_ALL) == 'y':            
            self.model.load_weights(weights_save_path)
      
    
