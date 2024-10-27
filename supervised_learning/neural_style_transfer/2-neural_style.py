#!/usr/bin/env python3
"""
Task 2
"""
import numpy as np
import tensorflow as tf


class NST:
    """
    This class performs tasks for Neural Style Transfer
    """
    style_layers = ['block1_conv1',
                    'block2_conv1',
                    'block3_conv1',
                    'block4_conv1',
                    'block5_conv1']
    content_layer = 'block5_conv2'

    def __init__(self, style_image, content_image, alpha=1e4, beta=1):
        """
        Class constructor for Neural Style Transfer class
        """
        error1 = "style_image must be a numpy.ndarray with shape (h, w, 3)"
        error2 = "content_image must be a numpy.ndarray with shape (h, w, 3)"
        if (not isinstance(style_image, np.ndarray) or style_image.ndim != 3
                or style_image.shape[-1] != 3):
            raise TypeError(error1)
        if (not isinstance(content_image, np.ndarray)
                or content_image.ndim != 3 or content_image.shape[-1] != 3):
            raise TypeError(error2)
        if not isinstance(alpha, (int, float)) or alpha < 0:
            raise TypeError("alpha must be a non-negative number")
        if not isinstance(beta, (int, float)) or beta < 0:
            raise TypeError("beta must be a non-negative number")

        self.style_image = self.scale_image(style_image)
        self.content_image = self.scale_image(content_image)
        self.alpha = alpha
        self.beta = beta
        self.load_model()

    @staticmethod
    def scale_image(image):
        """
        This method rescales an image such that its pixels values are between 0
        """
        error = "image must be a numpy.ndarray with shape (h, w, 3)"
        if (not isinstance(image, np.ndarray) or len(image.shape) != 3
                or image.shape[-1] != 3):
            raise TypeError(error)
        h, w, _ = image.shape
        if h > w:
            h_new = 512
            w_new = int(w * h_new / h)
        else:
            w_new = 512
            h_new = int(h * w_new / w)
        image = tf.convert_to_tensor(image, dtype=tf.float32)
        image = tf.image.resize(image, (h_new, w_new), method="bicubic")
        image = tf.clip_by_value(image / 255.0, 0, 1)
        return image[tf.newaxis, ...]

    def load_model(self):
        """
        This method loads the VGG19 model for Neural Style Transfer.
        """
        vgg = tf.keras.applications.VGG19(include_top=False,
                                          weights='imagenet')
        for layer in vgg.layers:
            if isinstance(layer, tf.keras.layers.MaxPooling2D):
                layer.__class__ = tf.keras.layers.AveragePooling2D
        vgg.trainable = False
        outputs = [vgg.get_layer(layer).output for layer in self.style_layers]
        outputs.append(vgg.get_layer(self.content_layer).output)
        self.model = tf.keras.models.Model(inputs=vgg.input, outputs=outputs)

    @staticmethod
    def gram_matrix(input_layer):
        """
        This method calculates the Gram matrices for a given layer.
        """
        error = "input_layer must be a tensor of rank 4"
        if (not isinstance(input_layer, (tf.Tensor, tf.Variable))
                or len(input_layer.shape) != 4):
            raise TypeError(error)

        _, h, w, c = input_layer.shape
        F = tf.reshape(input_layer, (h * w, c))
        gram = tf.matmul(F, F, transpose_a=True)
        gram = tf.expand_dims(gram, axis=0)

        nb_locations = tf.cast(h * w, tf.float32)

        return gram / nb_locations
