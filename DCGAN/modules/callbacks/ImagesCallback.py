
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import os



class ImagesCallback(keras.callbacks.Callback):

    def __init__(self, 
                 num_img    = 3, 
                 latent_dim = 100,
                 filename   = 'image-{epoch:03d}-{i:02d}.jpg',
                 run_dir    = './run'):
        self.num_img    = num_img
        self.latent_dim = latent_dim
        self.filename   = filename
        self.run_dir    = run_dir
        os.makedirs(run_dir, mode=0o750, exist_ok=True)


    def save_images(self, images, epoch):
        '''Save images as <filename>'''
        
        for i,image in enumerate(images):
            
            image = image.squeeze()  # Squeeze it if monochrome : (lx,ly,1) -> (lx,ly) 
        
            filenamei = self.run_dir+'/'+self.filename.format(epoch=epoch,i=i)
            
            if len(image.shape) == 2:
                plt.imsave(filenamei, image, cmap='gray_r')
            else:
                plt.imsave(filenamei, image)



    def on_epoch_end(self, epoch, logs=None):

        # ---- Get some points from latent space
        random_latent_vectors = tf.random.normal(shape=(self.num_img, self.latent_dim))

        # ---- Get fake images from generator
        generated_images = self.model.generator(random_latent_vectors)
        generated_images = np.array(generated_images)

        # ---- Save them
        self.save_images(generated_images, epoch)
