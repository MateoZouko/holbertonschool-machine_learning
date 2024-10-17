#!/usr/bin/env python3
"""
Task 0
"""

import tensorflow as tf
import numpy as np


class Yolo:
    """
    Yolo class
    """
    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """
        Constructor
        """
        self.model = tf.keras.models.load_model(model_path)
        with open(classes_path, 'r') as f:
            self.class_names = [line.strip() for line in f]
        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors

    def predict(self, image):
        """
        Prediction
        """
        input_size = (self.model.input.shape[1], self.model.input.shape[2])
        resized_image = tf.image.resize(image, input_size)
        resized_image = resized_image / 255.0  # Normalize the image

        outputs = self.model.predict(np.expand_dims(resized_image, axis=0))

        boxes, confidences, class_probs = self.process_outputs(outputs)
        
        return boxes, confidences, class_probs
    
    def process_outputs(self, outputs):
        """
        Process outputs
        """
        pass
