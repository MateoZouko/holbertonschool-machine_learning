#!/usr/bin/env python3
"""
Task 1
"""
from keras.models import load_model
import numpy as np
import math


class Yolo():
    """
    Yolo class
    """

    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        self.model = load_model(model_path)
        self.class_names = self.read_classes(classes_path)
        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors

    def read_classes(self, path):
        """
        Read the classes
        """

        list_class = []
        file = open(path, mode='r')
        classes = file.read().split("\n")
        for i in classes[0:-1]:
            list_class.append(i)

        return list_class

    def process_outputs(self, outputs, image_size):
        """
        Process the outputs
        """
        box_confidence = []
        boxes = []
        box_class_probs = []

        for i, output in enumerate(outputs):

            grid_height, grid_width, anchor_boxes, lastclasse = output.shape
            image_height, image_width = image_size[0], image_size[1]

            grid_x, grid_y = np.meshgrid(np.arange(grid_width),
                                         np.arange(grid_height))
            grid_x = grid_x.reshape(1, grid_height, grid_width, 1)
            grid_y = grid_y.reshape(1, grid_height, grid_width, 1)

            center_x = (1 / (1 + np.exp(-output[..., 0])) + grid_x)\
                / grid_width * image_width

            center_y = (1 / (1 + np.exp(-output[:, :, :, 1])) + grid_y)\
                / grid_height * image_height

            width = self.anchors[i][:, 0] * np.exp(output[:, :, :, 2])\
                / self.model.input.shape[1] * image_width

            height = self.anchors[i][:, 1] * np.exp(output[:, :, :, 3])\
                / self.model.input.shape[2] * image_height

            x1 = center_x - (width / 2)
            y1 = center_y - (height / 2)
            x2 = center_x + (width / 2)
            y2 = center_y + (height / 2)

            box = np.zeros((grid_height, grid_width, anchor_boxes, 4))
            box[:, :, :, 0] = x1
            box[:, :, :, 1] = y1
            box[:, :, :, 2] = x2
            box[:, :, :, 3] = y2
            boxes.append(box)

            confidence = 1 / (1 + np.exp(-output[:, :, :, 4]))
            confidence = confidence.reshape(grid_height, grid_width,
                                            anchor_boxes, 1)
            box_confidence.append(confidence)
            box_class_probs.append(1 / (1 + np.exp(-output[:, :, :, 5:])))

        return boxes, box_confidence, box_class_probs

    def filter_boxes(self, boxes, box_confidences, box_class_probs):
        """
        Filter the boxes
        """
        threshold = self.class_t
        selected_BB = []
        selected_conf = []
        selected_Class = []

        for nb_output in range(len(box_confidences)):
            grid_height, grid_width, anchor_boxes, lastclasse = \
                box_confidences[nb_output].shape

            for i in range(grid_height):
                for j in range(grid_width):
                    for k in range(anchor_boxes):

                        index_C = box_class_probs[nb_output][i, j, k]\
                            .argmax()

                        max_class = box_class_probs[nb_output][i, j, k].max()

                        if box_confidences[nb_output][i, j, k, 0]\
                                * max_class > threshold:
                            selected_BB.append(boxes[nb_output][i, j, k, 0:4])
                            selected_Class.append(index_C)

                            conf = box_confidences[nb_output][i, j, k]\
                                * box_class_probs[nb_output][i, j, k, index_C]
                            selected_conf.append(float(conf))

        selected_BB = np.array(selected_BB)
        selected_conf = np.array(selected_conf)
        selected_Class = np.array(selected_Class)
        return selected_BB, selected_Class, selected_conf

    def non_max_suppression(self, filtered_boxes, box_classes, box_scores):
        """
        Non max suppression
        """
        box_scores = box_scores.reshape(-1)
        box_classes = box_classes.reshape(-1)
        box_classes = box_classes.astype(int)
        box_classes = box_classes.tolist()
        box_scores = box_scores.tolist()
        filtered_boxes = filtered_boxes.reshape(-1, 4)
        new_boxes = []
        new_classes = []
        new_scores = []
        for i in range(len(box_scores)):
            if box_scores[i] > self.class_t:
                new_boxes.append(filtered_boxes[i])
                new_classes.append(box_classes[i])
                new_scores.append(box_scores[i])

        filtered_boxes = np.array(new_boxes)
        box_classes = np.array(new_classes)
        box_scores = np.array(new_scores)
        i = 0
        while i < len(filtered_boxes):
            j = i + 1
            while j < len(filtered_boxes):
                if self.iou(filtered_boxes[i], filtered_boxes[j]) > self.nms_t:
                    filtered_boxes = np.delete(filtered_boxes, j, axis=0)
                    box_scores = np.delete(box_scores, j, axis=0)
                    box_classes = np.delete(box_classes, j, axis=0)
                else:
                    j += 1
            i += 1
        return filtered_boxes, box_classes, box_scores
