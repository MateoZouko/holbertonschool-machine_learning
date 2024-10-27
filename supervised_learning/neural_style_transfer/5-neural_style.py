#!/usr/bin/env python3
"""
Task 5
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
        self.generate_features()

    @staticmethod
    def scale_image(image):
        """
        This method rescales an image such
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
        base_vgg = tf.keras.applications.VGG19(include_top=False,
                                               weights='imagenet')
        custom_object = {"MaxPooling2D": tf.keras.layers.AveragePooling2D}
        base_vgg.save("base_vgg")
        vgg = tf.keras.models.load_model("base_vgg",
                                         custom_objects=custom_object)
        vgg.trainable = False
        outputs = [vgg.get_layer(layer).output for layer in self.style_layers]
        outputs.append(vgg.get_layer(self.content_layer).output)
        self.model = tf.keras.models.Model(inputs=vgg.input, outputs=outputs)

    @staticmethod
    def gram_matrix(input_layer):
        """
        This method calculates the gram matrices for the input layer.
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

    def generate_features(self):
        """
        This method extracts the features used
        to calculate the neural style
        """
        content_image = tf.keras.applications.vgg19.preprocess_input(
            self.content_image * 255)
        style_image = tf.keras.applications.vgg19.preprocess_input(
            self.style_image * 255)
        style_outputs = self.model(style_image)
        content_outputs = self.model(content_image)
        self.gram_style_features = [self.gram_matrix(style_feature)
                                    for style_feature in style_outputs[:-1]]
        self.content_feature = content_outputs[-1]

    def layer_style_cost(self, style_output, gram_target):
        """
        This method calculates the style cost for a single layer.
        """
        error = "style_output must be a tensor of rank 4"
        if (not isinstance(style_output, (tf.Tensor, tf.Variable))
                or len(style_output.shape) != 4):
            raise TypeError(error)
        # last dimension of style_output
        c = style_output.shape[-1]
        error = f"gram_target must be a tensor of shape [1, {c}, {c}]"
        if (not isinstance(gram_target, (tf.Tensor, tf.Variable))
                or gram_target.shape != (1, c, c)):
            raise TypeError(error)
        gram_style = self.gram_matrix(style_output)
        return tf.reduce_mean(tf.square(gram_style - gram_target))

    def style_cost(self, style_outputs):
        """
        This method calculates the style cost for the generated image.
        """
        len_style_layers = len(self.style_layers)
        error = f"style_outputs must be a list with a length of \
{len_style_layers}"
        if (not isinstance(style_outputs, list)
                or len(style_outputs) != len_style_layers):
            raise TypeError(error)
        style_cost = 0
        for target, output in zip(self.gram_style_features, style_outputs):
            style_cost += self.layer_style_cost(output, target)
        return style_cost / len_style_layers
