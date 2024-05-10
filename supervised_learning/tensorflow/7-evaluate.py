#!/usr/bin/env python3
"""
6th task on tensoflow project
"""
import tensorflow.compat.v1 as tf


def evaluate(X, Y, save_path):
    """
    Function that evaluates the output of a neural network
    """
    saver = tf.train.import_meta_graph(save_path + '.meta')
    with tf.Session() as sess:
        saver.restore(sess, save_path)
        print("Modelo cargado desde:", save_path)

        x = tf.get_collection('x')[0]
        y = tf.get_collection('y')[0]

        pred = tf.get_collection('y_pred')[0]
        accuracy = tf.get_collection('accuracy')[0]
        loss = tf.get_collection('loss')[0]

        prediction_result, accuracy_result, loss_result = sess.run(
            [pred, accuracy, loss], feed_dict={x: X, y: Y})

        print("Predicciones:", prediction_result)
        print("Precisión:", accuracy_result)
        print("Pérdida:", loss_result)

        return prediction_result, accuracy_result, loss_result
