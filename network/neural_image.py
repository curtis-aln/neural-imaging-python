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
    def __init__(self):
        self.specific_path = ''
        if specific_media_to_train != "":
            media, name, self.is_training_images = self.load_specific_media()
            print(Fore.YELLOW + f"Sucesfully loaded the specified '{specific_media_to_train}' media")
            self.training_media = [media]
            self.training_names = [name]
        
        else:
            self.training_media, self.training_names, self.is_training_images = self.load_media_info_from_folder()
        
        # each media will have their own size and aspect ratio which needs to be fetched to create their own input space
        self.media_frame_sizes = get_media_shapes(self.training_media, self.is_training_images)
        print(Fore.YELLOW + f"{len(self.training_media)} instances found for training" + Style.RESET_ALL)

        # Flattening the data so that we can map it to the input space and feed into into the tensorflow trainer
        self.normalized_media = [normalize_and_reshape_media(media, self.is_training_images) for media in self.training_media]

        # creating the input data from the media sizes | automatically adjusts for videos
        if self.is_training_images:
            self.input_space = [create_input_data(size) for size in self.media_frame_sizes]
        else:
            self.input_space = [create_video_input_data(size, video.shape[0]) for size, video in zip(self.media_frame_sizes, self.training_media)]

        # creating the training datasets
        self.datasets = [self.create_dataset(inp, norm) for inp, norm in zip(self.input_space, self.normalized_media)]

        # loading and compiling the model using SIREN
        self.model = build_siren_model(config)
        print(Fore.GREEN + "model has been created" + Style.RESET_ALL)
        self.continue_training_model()            
        self.model.summary()
        
        
        self.model.compile(optimizer=model_optimizer, loss=model_loss)
        print(Fore.GREEN + f"model has been compiled with optimizer '{model_optimizer}' and loss '{model_loss}'" + Style.RESET_ALL)

        # at the end of training each model we store its best prediction of the training data here
        self.preditions = []
        
        # Important callbacks during training
        self.video_callback = VideoCallback(self, save_every=1, resolution=self.media_frame_sizes[0])

        
        print(Fore.GREEN + "Initialization Finished" + Style.RESET_ALL)

        
    def continue_training_model(self):
        if specific_media_to_train == '':
            return False

        # in the situation where there is already a model save file
        name = self.training_names[0]
        media_type = "image" if self.is_training_images else "video"
        path = model_save_folder_path + name + "_" + media_type + ".keras"
        file_exists = os.path.isfile(path)

        if not file_exists:
            return False
        
        print(Fore.BLUE + f"A model has been retrived from '{path}'")
        if input(Fore.BLUE + "would you like to continue training with this model? y/n " + Style.RESET_ALL) == 'y':            
            self.model.load_weights(path)
        return True
        
    

    def load_specific_media(self):
        name, extension = os.path.splitext(specific_media_to_train)
        extension = extension.lstrip('.')

        images = ["png", "jpeg"]
        videos = ["mp4", "wav"]
        is_training_images = extension in images

        if not (extension in images + videos):
            raise ValueError(f"Unsupported Filetype: {extension} ({specific_media_to_train})")
        
        media = None
        size = None
        if is_training_images:
            self.specific_path = image_dataset_path + specific_media_to_train
            media, size = load_image_from_file(self.specific_path, image_longest_length)
        else:
            self.specific_path = video_dataset_path + specific_media_to_train
            media, size = load_all_media_from_folder(self.specific_path, image_longest_length, frames_max)
        
        return media, name, is_training_images


    def load_media_info_from_folder(self):
        # loading the images and the videos from memory
        images, image_names = load_all_media_from_folder(image_dataset_path, image_longest_length, media_type='images')
        videos, video_names = load_all_media_from_folder(video_dataset_path, image_longest_length, media_type='videos', frame_count=frames_max)
        print(Fore.GREEN + "Images and/or VIdeos Sucessfully loaded to memory" + Style.RESET_ALL)

        # determining whether the media type is video or image
        is_training_images = False #len(images) != 0

        img_len, vid_len = len(images), len(videos) # todo
        if (img_len != 0 and vid_len != 0):
            print(Fore.MAGENTA + "There are both images and videos detected in the training folders, would you like videos (v) to be trained or images (i)? ")
            is_training_images = input(Fore.MAGENTA + ">>> " + Style.RESET_ALL) == "i"
        
        training_media, training_names = (images, image_names) if self.is_training_images else (videos, video_names)
        return training_media, training_names, is_training_images


    def train_model(self, hyper_resolution_during_training=False) -> tuple:   
        
        image_index = 0 # each model is trained one after the other
        for dataset, name, size in zip(self.datasets, self.training_names, self.media_frame_sizes):
            text = f"Training model '{name}' with training image shape {size} ({image_index}/{len(self.training_media)})"
            print(Fore.BLUE + text + Style.RESET_ALL)

            model_save_path = self.generate_model_save_path(name)
            results_save_path = self.generate_results_save_path(name)

            cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=model_save_path, 
                                                         verbose=1, mode='max', save_best_only=True)
            callbacks = [LossHistory(), SingleLineLogger(), cp_callback]

            if self.is_training_images:
                callbacks.append(self.video_callback)
            
            # this is needed for when media is being trained, we dont want the timelapse video to continue getting longer
            self.video_callback.reset(size, image_index)

            # exeption tells us the user wants to stop training | fitting the current dataset
            exeption = self.fit_image(dataset, callbacks, name)

            # now we tell the model to create its best prediction of what its supposed to draw
            prediction, _ = self.get_prediction(image_index, hyper_resolution_during_training)
            
            # now we need to save the prediction and prepare for the next media
            self.process_trained_model(image_index, prediction, model_save_path, results_save_path)
            
            if exeption:
                break

            image_index += 1
        
        quit()
        return self.preditions, self.media_frame_sizes, self.training_media
    
    
    def process_trained_model(self, image_index : int, prediction, model_save_path, results_save_path):
        # if images are being trained then we save the timelapse of it being generated
        if self.is_training_images:
            print(Fore.BLUE + f"saving timelapse to path '{results_save_path}'" + Style.RESET_ALL)
            self.video_callback.save_video(fps=timelapse_fps, output_path=results_save_path)

        # if its a video being trained we save just the video
        else:
            shape = self.training_media[image_index].shape
            save_flat_predictions_as_video(prediction, results_save_path, shape, video_predictions_fps)
        
        self.save_model(model_save_path)
        self.preditions.append(prediction) # for rendering later

    
    def generate_model_save_path(self, name=''):
        media_type = "image" if self.is_training_images else "video"
        extension = "keras"
        name = "model" if name == '' else name
        path = model_save_folder_path + name + "_" + media_type + "." + extension 
        return path
    
    def generate_results_save_path(self, name=''):
        folder = timelapse_save_path if self.is_training_images else final_predictions_save_path
        path = folder + name + ".mp4"
        return path


    def save_model(self, path):
        self.model.save(path)
        print(Fore.MAGENTA + f"model has been saved to folder path '{path}'" + Style.RESET_ALL)

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

