# ------------------------------------------------------------------
#     _____ _     _ _
#    |  ___(_) __| | | ___
#    | |_  | |/ _` | |/ _ \
#    |  _| | | (_| | |  __/
#    |_|   |_|\__,_|_|\___|                            DCGAN Example
# ------------------------------------------------------------------
# Formation Introduction au Deep Learning  (FIDLE)
# CNRS/SARI/DEVLOG 2020 - S. Arias, E. Maldonado, JL. Parouty
# ------------------------------------------------------------------
# by JL Parouty (dec 2020), based on François Chollet example
#
# Thanks to François Chollet example : https://keras.io/examples

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from IPython.display import display,Markdown
import os

class DCGAN(keras.Model):
    '''
    A DCGAN model, built from given generator and discriminator
    '''

    version = '1.0'



    def __init__(self, discriminator=None, generator=None, latent_dim=100, **kwargs):
        '''
        DCGAN instantiation with a given discriminator and generator
        args :
            discriminator : discriminator model
            generator : generator model
            latent_dim : latent space dimension
        return:
            None
        '''
        super(DCGAN, self).__init__(**kwargs)
        self.discriminator = discriminator
        self.generator     = generator
        self.latent_dim    = latent_dim
        print(f'Fidle DCGAN is ready :-)  latent dim = {latent_dim}')


        
    def call(self, inputs):
        '''
        Implementation of the model forward pass
        args:
            inputs : vectors from latent space
        return:
            output : Output of the generator
        '''
        outputs = self.generator(inputs)
        return outputs
                


    def compile(self, 
                discriminator_optimizer = keras.optimizers.Adam(), 
                generator_optimizer     = keras.optimizers.Adam(), 
                loss_function           = keras.losses.BinaryCrossentropy() ):
        super(DCGAN, self).compile()
        self.d_optimizer   = discriminator_optimizer
        self.g_optimizer   = generator_optimizer
        self.loss_fn       = loss_function
        self.d_loss_metric = keras.metrics.Mean(name="d_loss")
        self.g_loss_metric = keras.metrics.Mean(name="g_loss")



    @property
    def metrics(self):
        return [self.d_loss_metric, self.g_loss_metric]



    def train_step(self, inputs):
        '''
        Implementation of the training update.
        Receive some real images.
        This will compute loss, get gradients and update weights for generator and discriminator
        Return metrics.
        args:
            real_images : real images
        return:
            d_loss  : discriminator loss
            g_loss  : generator loss
        '''

        # ---- Prepare data for discriminator ----------------------
        # ----------------------------------------------------------
        #        
        # ---- Get the input we need, specified in the .fit()
        #      inputs is a tuple of tensors
        #
        if isinstance(inputs, tuple):
            real_images = inputs[0]
        else:
            real_images = inputs

        batch_size=tf.shape(real_images)[0]

        # Get some random points in the latent space
        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))

        # Generate fake images with the generator
        generated_images = self.generator(random_latent_vectors)

        # Combine them with real images
        combined_images = tf.concat([generated_images, real_images], axis=0)

        # Creation of labels corresponding to real or fake images
        labels = tf.concat(
            [tf.ones((batch_size, 1)), tf.zeros((batch_size, 1))], axis=0
        )
        # Add random noise to the labels - important trick!
        labels += 0.05 * tf.random.uniform(tf.shape(labels))

        # ---- Train the discriminator -----------------------------
        # ----------------------------------------------------------
        #
        # ---- Forward pass
        #      Run the forward pass and record operations with the GradientTape.
        #
        with tf.GradientTape() as tape:

            # Get predictions from discriminator 
            predictions = self.discriminator(combined_images)

            # Get loss
            d_loss = self.loss_fn(labels, predictions)

        # ---- Backward pass
        #      Retrieve gradients from gradient_tape and run one step
        #      of gradient descent to optimize trainable weights
        #
        grads = tape.gradient(d_loss, self.discriminator.trainable_weights)
        self.d_optimizer.apply_gradients( zip(grads, self.discriminator.trainable_weights) )

        # ---- Prepare data for generator ----------------------
        # ----------------------------------------------------------
        #
        # Sample random points in the latent space
        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))

        # Assemble labels that say "all real images"
        misleading_labels = tf.zeros((batch_size, 1))

        # ---- Train the generator ---------------------------------
        # ----------------------------------------------------------
        # We should *not* update the weights of the discriminator!
        #
        # ---- Forward pass
        #      Run the forward pass and record operations with the GradientTape.
        #
        with tf.GradientTape() as tape:

            # Get fake images from generator
            fake_images = self.generator(random_latent_vectors)

            # Get predictions from discriminator 
            predictions = self.discriminator(fake_images)

            # Get loss
            g_loss = self.loss_fn(misleading_labels, predictions)
        
        # ---- Backward pass (only for generator)
        #      Retrieve gradients from gradient_tape and run one step
        #      of gradient descent to optimize trainable weights
        #
        grads = tape.gradient(g_loss, self.generator.trainable_weights)
        self.g_optimizer.apply_gradients(zip(grads, self.generator.trainable_weights))

        # ---- Update and return metrics ---------------------------
        # ----------------------------------------------------------
        #
        self.d_loss_metric.update_state(d_loss)
        self.g_loss_metric.update_state(g_loss)

        return {
            "d_loss": self.d_loss_metric.result(),
            "g_loss": self.g_loss_metric.result(),
        }

    
    # def predict(self,inputs):
    #     '''Our predict function...'''
    #     z_mean, z_var, z  = self.encoder.predict(inputs)
    #     outputs           = self.decoder.predict(z)
    #     return outputs

        
    def save(self,filename):
        '''Save model in 2 part'''
        save_dir             = os.path.dirname(filename)
        filename, _extension = os.path.splitext(filename)
        # ---- Create directory if needed
        os.makedirs(save_dir, mode=0o750, exist_ok=True)
        # ---- Save models
        self.discriminator.save( f'{filename}-discriminator.h5' )
        self.generator.save(     f'{filename}-generator.h5'     )

    
    def reload(self,filename):
        '''Reload a 2 part saved model.
        Note : to train it, you need to .compile() it...'''
        filename, extension = os.path.splitext(filename)
        self.discriminator = keras.models.load_model(f'{filename}-discriminator.h5', compile=False)
        self.generator     = keras.models.load_model(f'{filename}-generator.h5'    , compile=False)
        print('Reloaded.')
                
        
    @classmethod
    def about(cls):
        '''Basic whoami method'''
        display(Markdown('<br>**FIDLE 2021 - DCGAN**'))
        print('Version              :', cls.version)
        print('TensorFlow version   :', tf.__version__)
        print('Keras version        :', tf.keras.__version__)
