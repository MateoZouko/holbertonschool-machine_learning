#!/usr/bin/env python3
"""
Task 1
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
        resized_image = resized_image / 255.0

        outputs = self.model.predict(np.expand_dims(resized_image, axis=0))

        boxes, confidences, class_probs = self.process_outputs(outputs)

        return boxes, confidences, class_probs

    def process_outputs(self, outputs, image_size):
        """
        Process outputs
        """
        boxes = []
        box_confidences = []
        box_class_probs = []

        image_height, image_width = image_size

        for output in outputs:
            grid_height, grid_width, anchor_boxes, _ = output.shape[:3]
            classes = output.shape[-1] - 5

            box = output[..., :4]
            box_confidence = output[..., 4:5]
            box_class_prob = output[..., 5:]

            for i in range(grid_height):
                for j in range(grid_width):
                    for k in range(anchor_boxes):
                        tx, ty, tw, th = box[i, j, k]

                        bx = (j + self.sigmoid(tx)) / grid_width
                        by = (i + self.sigmoid(ty)) / grid_height

                        bw = self.anchors[k][0] * np.exp(tw) / image_width
                        bh = self.anchors[k][1] * np.exp(th) / image_height

                        x1 = (bx - bw / 2) * image_width
                        y1 = (by - bh / 2) * image_height
                        x2 = (bx + bw / 2) * image_width
                        y2 = (by + bh / 2) * image_height

                        box[i, j, k] = [x1, y1, x2, y2]

            boxes.append(box)
            box_confidences.append(self.sigmoid(box_confidence))
            box_class_probs.append(self.sigmoid(box_class_prob))

        return (boxes, box_confidences, box_class_probs)

    @staticmethod
    def sigmoid(x):
        """
        Sigmoid function
        """
        return 1 / (1 + np.exp(-x))
