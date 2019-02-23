"""
Implementation of `3.1 Appearance-based Gaze Estimation` from
[Learning from Simulated and Unsupervised Images through Adversarial Training](https://arxiv.org/pdf/1612.07828v1.pdf).
"""

from __future__ import print_function, division
import keras.backend as K
from keras.optimizers import RMSprop
from keras.layers.merge import _Merge
from functools import partial
import os
import sys
import keras
from keras import applications
from keras import layers
from keras import models
from keras import optimizers
from keras.preprocessing import image
import numpy as np
import tensorflow as tf
import h5py
import plot_images_compare
import plot_images_compare_2
import plot_loss
from utils.image_history_buffer import ImageHistoryBuffer

class RandomWeightedAverage(_Merge):
    """Provides a (random) weighted average between real and generated image samples"""
    def _merge_function(self, inputs):
        alpha = K.random_uniform((512, 1, 1, 1))
        return (alpha * inputs[0]) + ((1 - alpha) * inputs[1])

# directories
print('tf-version',tf.__version__, 'keras-version', keras.__version__)

MP2GAZE_DATA_FNAME = 'gaze.npz'
SYN_DATA_FNAME = 'syn.npz'

path = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join('.', 'input')
cache_dir = os.path.join(path, 'cache')

# load the data file and extract dimensions
data_size = 200000
gaze_data = np.load(os.path.join(data_dir, MP2GAZE_DATA_FNAME))
real_image_stack = gaze_data['real']
real_image_stack = np.reshape(real_image_stack, newshape=real_image_stack.shape + (1,))
if data_size > np.size(real_image_stack, 0):
    data_size = np.size(real_image_stack, 0)
real_image_stack = real_image_stack[:data_size]

unit_data = np.load(os.path.join(data_dir, SYN_DATA_FNAME))
syn_image_stack = unit_data['syn']
syn_image_stack = np.reshape(syn_image_stack, newshape=syn_image_stack.shape + (1,))
if data_size > np.size(syn_image_stack, 0):
    data_size = np.size(syn_image_stack, 0)
syn_image_stack = syn_image_stack[:data_size]

sample_images = syn_image_stack[:2]

img_width = 60
img_height = 36
img_channels = 1

# training params
nb_steps = 10000
batch_size = 512
k_d = 1  # number of discriminator updates per step
k_g = 2  # number of generative network updates per step
log_interval = 50

def refiner_network(input_image_tensor):
    """
    The refiner network, Rθ, is a residual network (ResNet). It modifies the synthetic image on a pixel level, rather
    than holistically modifying the image content, preserving the global structure and annotations.

    :param input_image_tensor: Input tensor that corresponds to a synthetic image.
    :return: Output tensor that corresponds to a refined synthetic image.
    """
    def resnet_block(input_features, nb_features=64, nb_kernel_rows=3, nb_kernel_cols=3):
        """
        A ResNet block with two `nb_kernel_rows` x `nb_kernel_cols` convolutional layers,
        each with `nb_features` feature maps.

        See Figure 6 in https://arxiv.org/pdf/1612.07828v1.pdf.

        :param input_features: Input tensor to ResNet block.
        :return: Output tensor from ResNet block.
        """
        y = layers.Convolution2D(nb_features, nb_kernel_rows, nb_kernel_cols, border_mode='same')(input_features)
        y = layers.Activation('relu')(y)
        y = layers.Convolution2D(nb_features, nb_kernel_rows, nb_kernel_cols, border_mode='same')(y)

        y = layers.add([input_features, y])
        return layers.Activation('relu')(y)

    # an input image of size w × h is convolved with 3 × 3 filters that output 64 feature maps
    x = layers.Convolution2D(64, 3, 3, border_mode='same', activation='relu')(input_image_tensor)

    # the output is passed through 4 ResNet blocks
    for _ in range(4):
        x = resnet_block(x)

    # the output of the last ResNet block is passed to a 1 × 1 convolutional layer producing 1 feature map
    # corresponding to the refined synthetic image
    return layers.Convolution2D(img_channels, 1, 1, border_mode='same', activation='tanh')(x)

def discriminator_network(input_image_tensor):
    """
    The discriminator network, Dφ, contains 5 convolution layers and 2 max-pooling layers.

    :param input_image_tensor: Input tensor corresponding to an image, either real or refined.
    :return: Output tensor that corresponds to the probability of whether an image is real or refined.
    """
    x = layers.Convolution2D(96, 3, 3, border_mode='same', subsample=(2, 2), activation='relu')(input_image_tensor)
    x = layers.Convolution2D(64, 3, 3, border_mode='same', subsample=(2, 2), activation='relu')(x)
    x = layers.MaxPooling2D(pool_size=(3, 3), border_mode='same', strides=(1, 1))(x)
    x = layers.Convolution2D(32, 3, 3, border_mode='same', subsample=(1, 1), activation='relu')(x)
    x = layers.Convolution2D(32, 1, 1, border_mode='same', subsample=(1, 1), activation='relu')(x)
    x = layers.Convolution2D(2, 1, 1, border_mode='same', subsample=(1, 1), activation='relu')(x)

    # here one feature map corresponds to `is_real` and the other to `is_refined`,
    # and the custom loss function is then `tf.nn.sparse_softmax_cross_entropy_with_logits`
    return layers.Reshape((-1, 2))(x)

def adversarial_training(WLoss_AdverLoss = "W", refiner_model_path=None, discriminator_model_path=None):
    refiner_cp = 0
    discriminator_cp = 0
    if WLoss_AdverLoss == "W":
        print("--------Running with Wasserstein Loss------")
        """Adversarial training of refiner network Rθ and discriminator network Dφ."""
        # define model input and output tensors
        real_image_tensor = layers.Input(shape=(img_height, img_width, img_channels))

        synthetic_image_tensor = layers.Input(shape=(img_height, img_width, img_channels))
        refined_image_tensor = refiner_network(synthetic_image_tensor)

        refined_or_real_image_tensor = layers.Input(shape=(img_height, img_width, img_channels))
        discriminator_output = discriminator_network(refined_or_real_image_tensor)
        # define models
        # Construct weighted average between real synthetic and synthetic images
        interpolated_img = RandomWeightedAverage()([real_image_tensor, refined_or_real_image_tensor])
        validity_interp = discriminator_network(interpolated_img)

        refiner_model = models.Model(input=synthetic_image_tensor, output=refined_image_tensor, name='refiner')
        discriminator_model = models.Model(input=[real_image_tensor,refined_or_real_image_tensor], outputs=[discriminator_output,validity_interp],
                                           name='discriminator')

        # combined must output the refined image along w/ the disc's classification of it for the refiner's self-reg loss
        refiner_model_output = refiner_model(synthetic_image_tensor)
        [combined_output1,combined_output2] = discriminator_model([real_image_tensor,refiner_model_output])
        combined_model = models.Model(input=[real_image_tensor,synthetic_image_tensor], outputs=[refiner_model_output, combined_output1],name='combined')

        discriminator_model_output_shape = discriminator_model.output_shape

        print(refiner_model.summary())
        print(discriminator_model.summary())
        print(combined_model.summary())
    else:
        print("--------Running with adverserial Loss------")
        synthetic_image_tensor = layers.Input(shape=(img_height, img_width, img_channels))
        refined_image_tensor = refiner_network(synthetic_image_tensor)

        refined_or_real_image_tensor = layers.Input(shape=(img_height, img_width, img_channels))
        discriminator_output = discriminator_network(refined_or_real_image_tensor)
        # define modelS
        refiner_model = models.Model(input=synthetic_image_tensor, output=refined_image_tensor, name='refiner')
        discriminator_model = models.Model(input=refined_or_real_image_tensor, output=discriminator_output,
                                           name='discriminator')
        # combined must output the refined image along w/ the disc's classification of it for the refiner's self-reg loss
        refiner_model_output = refiner_model(synthetic_image_tensor)
        combined_output = discriminator_model(refiner_model_output)
        combined_model = models.Model(input=synthetic_image_tensor, output=[refiner_model_output, combined_output],
                                      name='combined')
        discriminator_model_output_shape = discriminator_model.output_shape
        print(refiner_model.summary())
        print(discriminator_model.summary())
        print(combined_model.summary())

    def gradient_penalty_loss(y_true, y_pred, averaged_samples):
        """
        Computes gradient penalty based on prediction and weighted real / fake samples
        """
        gradients = K.gradients(y_pred, averaged_samples)[0]
        # compute the euclidean norm by squaring ...
        gradients_sqr = K.square(gradients)
        #   ... summing over the rows ...
        gradients_sqr_sum = K.sum(gradients_sqr,
                                  axis=np.arange(1, len(gradients_sqr.shape)))
        #   ... and sqrt
        gradient_l2_norm = K.sqrt(gradients_sqr_sum)
        # compute lambda * (1 - ||grad||)^2 still for each single sample
        gradient_penalty = K.square(1 - gradient_l2_norm)
        # return the mean as loss over all the batch samples
        return K.mean(gradient_penalty)

    def wasserstein_loss( y_true, y_pred):
        y_true = tf.reshape(y_true, (-1, 2))
        y_pred = tf.reshape(y_pred, (-1, 2))
        return K.mean(y_true * y_pred)

    # define custom l1 loss function for the refiner
    def self_regularization_loss(y_true, y_pred):
        delta = 0.0001  # FIXME: need to figure out an appropriate value for this
        return tf.multiply(delta, tf.reduce_sum(tf.abs(y_pred - y_true)))

    # define custom local adversarial loss (softmax for each image section) for the discriminator
    # the adversarial loss function is the sum of the cross-entropy losses over the local patches
    def local_adversarial_loss(y_true, y_pred):
        # y_true and y_pred have shape (batch_size, # of local patches, 2), but really we just want to average over
        # the local patches and batch size so we can reshape to (batch_size * # of local patches, 2)
        y_true = tf.reshape(y_true, (-1, 2))
        y_pred = tf.reshape(y_pred, (-1, 2))
        loss = tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y_pred)
        return tf.reduce_mean(loss)

    # compile models
    if WLoss_AdverLoss == "W":
        optimizer = RMSprop(lr=0.00005)
        # Use Python partial to provide loss function with additional
        # 'averaged_samples' argument
        partial_gp_loss = partial(gradient_penalty_loss,
                                  averaged_samples=interpolated_img)
        partial_gp_loss.__name__ = 'gradient_penalty'  # Keras requires function names

        refiner_model.compile(optimizer=optimizer, loss=self_regularization_loss)
        discriminator_model.compile(optimizer=optimizer, loss=[wasserstein_loss,
                                                         partial_gp_loss],loss_weights=[1, 10])
        discriminator_model.trainable = False
        combined_model.compile(optimizer=optimizer, loss=[self_regularization_loss, wasserstein_loss])
    else:
        sgd = optimizers.SGD(lr=0.001)

        refiner_model.compile(optimizer=sgd, loss=self_regularization_loss)
        discriminator_model.compile(optimizer=sgd, loss=local_adversarial_loss)
        discriminator_model.trainable = False
        combined_model.compile(optimizer=sgd, loss=[self_regularization_loss, local_adversarial_loss])

    # data generators
    datagen = image.ImageDataGenerator(
        preprocessing_function=applications.xception.preprocess_input,
        data_format='channels_last')

    flow_from_directory_params = {'target_size': (img_height, img_width),
                                  'color_mode': 'grayscale' if img_channels == 1 else 'rgb',
                                  'class_mode': None,
                                  'batch_size': batch_size}
    flow_params = {'batch_size': batch_size}

    synthetic_generator = datagen.flow(
        x=syn_image_stack,
        **flow_params
    )
    sample_generator = datagen.flow(
        x=sample_images,
        **flow_params
    )
    samples = sample_generator.next()

    real_generator = datagen.flow(
        x=real_image_stack,
        **flow_params
    )

    def get_image_batch(generator):
        """keras generators may generate an incomplete batch for the last batch"""
        img_batch = generator.next()
        if len(img_batch) != batch_size:
            img_batch = generator.next()

        assert len(img_batch) == batch_size

        return img_batch

    # Adversarial ground truths
    # the target labels for the cross-entropy loss layer are 0 for every yj (real) and 1 for every xi (refined)
    if WLoss_AdverLoss =="W":
        y_real = np.array([[[1.0, 0.0]] * discriminator_model_output_shape[1][1]] * batch_size)
        y_refined = np.array([[[0.0, 1.0]] * discriminator_model_output_shape[1][1]] * batch_size)
        y_dummy = np.array([[[0.0, 0.0]] * discriminator_model_output_shape[1][1]] * batch_size)
        assert y_real.shape == (batch_size, discriminator_model_output_shape[1][1], 2)
    else:
        # the target labels for the cross-entropy loss layer are 0 for every yj (real) and 1 for every xi (refined)
        y_real = np.array([[[1.0, 0.0]] * discriminator_model_output_shape[1]] * batch_size)
        y_refined = np.array([[[0.0, 1.0]] * discriminator_model_output_shape[1]] * batch_size)
        assert y_real.shape == (batch_size, discriminator_model_output_shape[1], 2)

    if not refiner_model_path:
        # we first train the Rθ network with just self-regularization loss for 1,000 steps
        print('pre-training the refiner network...')
        gen_loss = np.zeros(shape=len(refiner_model.metrics_names))
        gen_loss_pre_training_vec = list()
        if WLoss_AdverLoss == "W":
            folder_name = 'pre training refiner loss wasserstein'
            plot_name = 'pre-training refiner loss wasserstein.png'

        else:
            folder_name = 'pre training refiner loss original'
            plot_name = 'pre-training refiner loss.png'

        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
        for i in range(1000):
            synthetic_image_batch = get_image_batch(synthetic_generator)
            gen_loss = np.add(refiner_model.train_on_batch(synthetic_image_batch, synthetic_image_batch), gen_loss)

            if not i % log_interval and i != 0:
                sub_folder_name = 'compare_images_batch_train_step_{}'.format(i)
                sub_folder_name2 = 'compare_same_images'

                print('Saving batch of refined images during pre-training at step: {}'.format(i))

                synthetic_image_batch = get_image_batch(synthetic_generator)
                refined_image_batch = refiner_model.predict_on_batch(synthetic_image_batch)
                refined_images_compare = refiner_model.predict_on_batch(samples)
                plot_images_compare_2.plot_compare2(samples, refined_images_compare,
                                                    os.path.join(folder_name, sub_folder_name2), i)
                plot_images_compare.plot_compare(synthetic_image_batch, refined_image_batch,
                                                 os.path.join(folder_name, sub_folder_name), number_of_plots=2)
                gen_loss_pre_training_vec.append(gen_loss)
                plot_loss.plot_loss_vec(gen_loss_pre_training_vec, plot_name)

                print('Refiner model self regularization loss: {}.'.format(gen_loss / log_interval))
                gen_loss = np.zeros(shape=len(refiner_model.metrics_names))
        refiner_model.save(os.path.join(cache_dir, 'refiner_model_pre_trained.h5'))
    else:
        refiner_model.load_weights(refiner_model_path)
        if "pre" not in refiner_model_path:
            refiner_cp = int(((refiner_model_path.split('_step_'))[1].split('.h5'))[0])
        else:
            refiner_cp = 0
        print(refiner_cp)

    if not discriminator_model_path:
        # and Dφ for 200 steps (one mini-batch for refined images, another for real)
        print('pre-training the discriminator network...')
        disc_loss = np.zeros(shape=len(discriminator_model.metrics_names))
        disc_loss_pre_training_vec = list()
        for _ in range(100):
            real_image_batch = get_image_batch(real_generator)
            if WLoss_AdverLoss == "W":
                plot_name = 'pre-training discriminator_model_path loss wasserstein.png'
                disc_loss_pre = discriminator_model.train_on_batch([real_image_batch,real_image_batch], [y_real,y_dummy])
                disc_loss = np.add(disc_loss_pre[0] , disc_loss)
                
                synthetic_image_batch = get_image_batch(synthetic_generator)
                refined_image_batch = refiner_model.predict_on_batch(synthetic_image_batch)
                disc_loss_pre = discriminator_model.train_on_batch([real_image_batch,refined_image_batch], [y_refined,y_dummy])
                disc_loss =  np.add(disc_loss_pre[0], disc_loss)
            else:
                plot_name = 'pre-training discriminator_model_path loss.png'
                disc_loss = np.add(discriminator_model.train_on_batch(real_image_batch, y_real), disc_loss)

                synthetic_image_batch = get_image_batch(synthetic_generator)
                refined_image_batch = refiner_model.predict_on_batch(synthetic_image_batch)
                disc_loss = np.add(discriminator_model.train_on_batch(refined_image_batch, y_refined), disc_loss)
            
            disc_loss_pre_training_vec.append(disc_loss)
            plot_loss.plot_loss_vec(disc_loss_pre_training_vec, plot_name)
            
        discriminator_model.save(os.path.join(cache_dir, 'discriminator_model_pre_trained.h5'))
        print('Discriminator model loss: {}.'.format(disc_loss / (100 * 2)))
    else:
        discriminator_model.load_weights(discriminator_model_path)

    image_history_buffer = ImageHistoryBuffer((0, img_height, img_width, img_channels), batch_size * 100, batch_size)

    combined_loss = np.zeros(shape=len(combined_model.metrics_names))
    disc_loss_real = np.zeros(shape=len(discriminator_model.metrics_names))
    disc_loss_refined = np.zeros(shape=len(discriminator_model.metrics_names))
    if WLoss_AdverLoss == "W":
        folder_name = 'train_loss_wasserstein'
        plot_name1 = 'training refiner loss_w.png'
        plot_name2 = 'real training discriminator loss_w.png'
        plot_name3 = 'refined training discriminator loss_w.png'
    else:
        folder_name = 'train_loss_original'
        plot_name1 = 'training refiner loss_a.png'
        plot_name2 = 'real training discriminator loss_a.png'
        plot_name3 = 'refined training discriminator loss_a.png'
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    # see Algorithm 1 in https://arxiv.org/pdf/1612.07828v1.pdf
    refiner_loss_training_vec = list()
    disc_loss_real_training_vec = list()
    disc_loss_syn_training_vec = list()

    for i in range(refiner_cp , nb_steps):
        # train the refiner
        for _ in range(k_g * 2):
            # sample a mini-batch of synthetic images
            synthetic_image_batch = get_image_batch(synthetic_generator)

            # update θ by taking an SGD step on mini-batch loss LR(θ)
            if WLoss_AdverLoss == "W":
                combined_loss = np.add(combined_model.train_on_batch([synthetic_image_batch, synthetic_image_batch],
                                                                 [synthetic_image_batch, y_real]), combined_loss)
            else:
                combined_loss = np.add(combined_model.train_on_batch(synthetic_image_batch,
                                                                 [synthetic_image_batch, y_real]), combined_loss)

        for _ in range(k_d):
            if WLoss_AdverLoss == "W":
                # sample a mini-batch of synthetic and real images
                synthetic_image_batch = get_image_batch(synthetic_generator)
                real_image_batch = get_image_batch(real_generator)

                # refine the synthetic images w/ the current refiner
                refined_image_batch = refiner_model.predict_on_batch(synthetic_image_batch)

                # use a history of refined images
                half_batch_from_image_history = image_history_buffer.get_from_image_history_buffer()
                image_history_buffer.add_to_image_history_buffer(refined_image_batch)

                if len(half_batch_from_image_history):
                    refined_image_batch[:batch_size // 2] = half_batch_from_image_history

                # update φ by taking an SGD step on mini-batch loss LD(φ)
                disc_loss_pre = discriminator_model.train_on_batch([real_image_batch,real_image_batch], [y_real,y_dummy])
                disc_loss_real = np.add(disc_loss_pre[0], disc_loss_real)
                disc_loss_pre = discriminator_model.train_on_batch([real_image_batch, refined_image_batch], [y_refined,y_dummy])
                disc_loss_refined = np.add(disc_loss_pre[0],disc_loss_refined)
            else:
                synthetic_image_batch = get_image_batch(synthetic_generator)
                real_image_batch = get_image_batch(real_generator)

                # refine the synthetic images w/ the current refiner
                refined_image_batch = refiner_model.predict_on_batch(synthetic_image_batch)

                # use a history of refined images
                half_batch_from_image_history = image_history_buffer.get_from_image_history_buffer()
                image_history_buffer.add_to_image_history_buffer(refined_image_batch)

                if len(half_batch_from_image_history):
                    refined_image_batch[:batch_size // 2] = half_batch_from_image_history

                # update φ by taking an SGD step on mini-batch loss LD(φ)
                disc_loss_real = np.add(discriminator_model.train_on_batch(real_image_batch, y_real), disc_loss_real)
                disc_loss_refined = np.add(discriminator_model.train_on_batch(refined_image_batch, y_refined),
                                           disc_loss_refined)
        
        if not i % log_interval and i != 0:
            # plot batch of refined images w/ current refiner
            sub_folder_name = 'refined_image_batch_step_{}'.format(i)
            sub_folder_name2 = 'compare_same_images'
            print('Saving batch of refined images at adversarial step: {}.'.format(i))

            synthetic_image_batch = get_image_batch(synthetic_generator)
            refined_image_batch = refiner_model.predict_on_batch(synthetic_image_batch)

            refined_images_compare = refiner_model.predict_on_batch(samples)
            plot_images_compare_2.plot_compare2(samples, refined_images_compare,
                                                os.path.join(folder_name, sub_folder_name2), i)
            plot_images_compare.plot_compare(synthetic_image_batch, refined_image_batch,
                                             os.path.join(folder_name, sub_folder_name), number_of_plots=5)

            # log loss summary
            print('Refiner model loss: {}.'.format(combined_loss / (log_interval * k_g * 2)))
            print('Discriminator model loss real: {}.'.format(disc_loss_real / (log_interval * k_d * 2)))
            print('Discriminator model loss refined: {}.'.format(disc_loss_refined / (log_interval * k_d * 2)))
            refiner_loss_training_vec.append(combined_loss[1])
            disc_loss_real_training_vec.append(disc_loss_real[0])
            disc_loss_syn_training_vec.append(disc_loss_refined[0])
            plot_loss.plot_loss_vec(refiner_loss_training_vec, plot_name1)
            plot_loss.plot_loss_vec(disc_loss_real_training_vec, plot_name2)
            plot_loss.plot_loss_vec(disc_loss_syn_training_vec, plot_name3)

            combined_loss = np.zeros(shape=len(combined_model.metrics_names))
            disc_loss_real = np.zeros(shape=len(discriminator_model.metrics_names))
            disc_loss_refined = np.zeros(shape=len(discriminator_model.metrics_names))

            # save model checkpoints
            if WLoss_AdverLoss == "W":
                model_checkpoint_base_name = os.path.join(cache_dir, '{}_model_w_step_{}.h5')
            else:
                model_checkpoint_base_name = os.path.join(cache_dir, '{}_model_a_step_{}.h5')
            refiner_model.save(model_checkpoint_base_name.format('refiner', i))
            discriminator_model.save(model_checkpoint_base_name.format('discriminator', i))

def main(W_A, refiner_model_path, discriminator_model_path):
    adversarial_training(W_A, refiner_model_path, discriminator_model_path)


if __name__ == '__main__':
    # TODO: if pre-trained models are passed in, we don't take the steps they've been trained for into account
    W_A = sys.argv[1]
    # W_A = 'A'
    refiner_model_path = sys.argv[2] if len(sys.argv) >= 3 else None
    discriminator_model_path = sys.argv[3] if len(sys.argv) >= 4 else None
    # refiner_model_path = None
    # discriminator_model_path = None
    main(W_A, refiner_model_path, discriminator_model_path)