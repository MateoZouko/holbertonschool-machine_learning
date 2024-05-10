#!/usr/bin/env python3
"""
Task 6
"""

import tensorflow.compat.v1 as tf

calculate_accuracy = __import__('3-calculate_accuracy').calculate_accuracy
calculate_loss = __import__('4-calculate_loss').calculate_loss
create_placeholders = __import__('0-create_placeholders').create_placeholders
forward_prop = __import__('2-forward_prop').forward_prop
create_train_op = __import__('5-create_train_op').create_train_op


def train(X_train, Y_train, X_valid, Y_valid, layer_sizes, activations,
          alpha, iterations, save_path="/tmp/model.ckpt"):
    """
    Function that builds, trains, and saves a neural network classifier
    """

    x, y = create_placeholders(X_train.shape[1], Y_train.shape[1])

    pred = forward_prop(x, layer_sizes, activations)

    loss = calculate_loss(y, pred)
    accuracy = calculate_accuracy(y, pred)

    train_op = create_train_op(loss, alpha)

    tf.add_to_collection('x', x)
    tf.add_to_collection('y', y)
    tf.add_to_collection('y_pred', pred)
    tf.add_to_collection('loss', loss)
    tf.add_to_collection('accuracy', accuracy)
    tf.add_to_collection('train_op', train_op)

    init = tf.global_variables_initializer()

    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(init)

        for i in range(iterations):
            _, train_cost, train_acc = sess.run([train_op, loss, accuracy],
                                                feed_={x: X_train, y: Y_train})
            valid_cost, valid_acc = sess.run([loss, accuracy],
                                             feed_={x: X_valid, y: Y_valid})

            if i % 100 == 0 or i == iterations - 1:
                print("After {} iterations:".format(i))
                print("\tTraining Cost: {}".format(train_cost))
                print("\tTraining Accuracy: {}".format(train_acc))
                print("\tValidation Cost: {}".format(valid_cost))
                print("\tValidation Accuracy: {}".format(valid_acc))

        saver.save(sess, save_path)

    return save_path
