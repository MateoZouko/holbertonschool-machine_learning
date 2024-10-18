#!/usr/bin/env python3
"""
Task 3
"""
import keras as K
import numpy as np
import tensorflow as tf
import os
import cv2


class Yolo:
    """
    Yolo class for object detection.
    """
    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """
        Constructor.
        """
        self.model = K.models.load_model(model_path)
        with open(classes_path, 'r') as classes:
            self.class_names = [line.strip() for line in classes]
        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors

    def sigmoid(self, x):
        """
        Sigmoid function.
        """
        return 1 / (1 + np.exp(-x))

    def process_outputs(self, outputs, image_size):
        """
        Process Darknet outputs.
        """
        boxes = []
        box_confidences = []
        box_class_probs = []
        image_height, image_width = image_size

        for i, output in enumerate(outputs):
            grid_height, grid_width, _, _ = output.shape

            box_confidence = self.sigmoid(output[..., 4:5])
            box_class_prob = self.sigmoid(output[..., 5:])

            box_confidences.append(box_confidence)
            box_class_probs.append(box_class_prob)

            tx = output[..., 0]
            ty = output[..., 1]
            tw = output[..., 2]
            th = output[..., 3]

            anchor_w = self.anchors[i, :, 0]
            anchor_h = self.anchors[i, :, 1]
            cx, cy = np.meshgrid(np.arange(grid_width),
                                 np.arange(grid_height))

            cx = np.expand_dims(cx, axis=-1)
            cy = np.expand_dims(cy, axis=-1)

            bx = (self.sigmoid(tx) + cx) / grid_width
            by = (self.sigmoid(ty) + cy) / grid_height
            bw = (np.exp(tw) * anchor_w) / self.model.input.shape[1]
            bh = (np.exp(th) * anchor_h) / self.model.input.shape[2]

            x1 = (bx - bw / 2) * image_width
            y1 = (by - bh / 2) * image_height
            x2 = (bx + bw / 2) * image_width
            y2 = (by + bh / 2) * image_height

            box = np.stack([x1, y1, x2, y2], axis=-1)
            boxes.append(box)

        return boxes, box_confidences, box_class_probs

    def filter_boxes(self, boxes, box_confidences, box_class_probs):
        """
        Filter boxes.
        """
        filtered_boxes = []
        box_classes = []
        box_scores = []

        for i in range(len(boxes)):
            box_confidence = box_confidences[i]
            box_class_prob = box_class_probs[i]

            box_scores_level = box_confidence * box_class_prob

            box_class_argmax = np.argmax(box_scores_level, axis=-1)
            box_class_score = np.max(box_scores_level, axis=-1)

            mask = box_class_score >= self.class_t

            filtered_boxes.append(boxes[i][mask])
            box_classes.append(box_class_argmax[mask])
            box_scores.append(box_class_score[mask])

        filtered_boxes = np.concatenate(filtered_boxes)
        box_classes = np.concatenate(box_classes)
        box_scores = np.concatenate(box_scores)

        return filtered_boxes, box_classes, box_scores

    def nms(self, boxes, scores, iou_threshold):
        """
        Non-max suppression.
        """

        boxes_tensor = tf.convert_to_tensor(boxes, dtype=tf.float32)
        scores_tensor = tf.convert_to_tensor(scores, dtype=tf.float32)

        selected_indices = tf.image.non_max_suppression(
            boxes_tensor, scores_tensor, max_output_size=boxes.shape[0],
            iou_threshold=iou_threshold)

        return selected_indices.numpy()

    def non_max_suppression(self, filtered_boxes, box_classes, box_scores):
        """
        Suppresses all non-max filter boxes to return predicted bounding boxes.
        """
        unique_classes = np.unique(box_classes)

        box_predictions = []
        predicted_box_classes = []
        predicted_box_scores = []

        for cls in unique_classes:
            mask = np.where(box_classes == cls)
            boxes_of_class = filtered_boxes[mask]
            class_scores = box_scores[mask]
            class_classes = box_classes[mask]

            keep = self.nms(boxes_of_class, class_scores, self.nms_t)

            box_predictions.append(boxes_of_class[keep])
            predicted_box_classes.append(class_classes[keep])
            predicted_box_scores.append(class_scores[keep])

        box_predictions = np.concatenate(box_predictions, axis=0)
        predicted_box_classes = np.concatenate(predicted_box_classes, axis=0)
        predicted_box_scores = np.concatenate(predicted_box_scores, axis=0)

        return box_predictions, predicted_box_classes, predicted_box_scores

    @staticmethod
    def load_images(folder_path):
        """
        Load images.
        """
        images = []
        images_paths = []

        for filename in os.listdir(folder_path):
            path = folder_path + '/' + filename
            image = cv2.imread(path)
            images.append(image)
            images_paths.append(path)

        return images, images_paths

    def preprocess_images(self, images):
        """
        Preprocess images.
        """
        pimages = []
        image_shapes = []

        for image in images:
            image_shapes.append(image.shape[:2])
            pimage = cv2.resize(image, (416, 416),
                                interpolation=cv2.INTER_CUBIC)
            pimage = pimage / 255
            pimage = pimage.astype(np.float32)
            pimages.append(pimage)

        pimages = np.array(pimages)
        image_shapes = np.array(image_shapes)

        return pimages, image_shapes
