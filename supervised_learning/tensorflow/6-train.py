#!/usr/bin/env python3
"""
6th task on tensoflow project
"""
import tensorflow.compat.v1 as tf

calculate_accuracy = __import__('3-calculate_accuracy').calculate_accuracy
calculate_loss = __import__('4-calculate_loss').calculate_loss
create_placeholders = __import__('0-create_placeholders').create_placeholders
create_train_op = __import__('5-create_train_op').create_train_op
forward_prop = __import__('2-forward_prop').forward_prop


def train(X_train, Y_train, X_valid, Y_valid, layer_sizes, activations,
          alpha, iterations, save_path="/tmp/model.ckpt"):
    """
    Function that builds, trains, and saves a neural network
    """
    nx = X_train.shape[1]
    classes = Y_train.shape[1]

    X_pl, Y_pl = create_placeholders(nx, classes)

    Y_pred = forward_prop(X_pl, layer_sizes, activations)
    loss = calculate_loss(Y_pl, Y_pred)
    accuracy = calculate_accuracy(Y_pl, Y_pred)
    train_step = create_train_op(loss, alpha)

    init = tf.initialize_all_variables()

    sess = tf.Session()
    sess.run(init)

    for i in range(iterations + 1):

        train_data = {X_pl: X_train, Y_pl: Y_train}
        a, c = sess.run([accuracy, loss], feed_dict=train_data)

        test_data = {X_pl: X_valid, Y_pl: Y_valid}

        test_a, test_c = sess.run([accuracy, loss], feed_dict=test_data)

        if i < iterations:
            sess.run(train_step, feed_dict=train_data)

        if i == 0 or i == iterations or i % 100 == 0:
            print("After {} iterations:".format(i))
            print("\tTraining Cost: {}".format(c))
            print("\tTraining Accuracy: {}".format(a))
            print("\tValidation Cost: {}".format(test_c))
            print("\tValidation Accuracy: {}".format(test_a))

    saver = tf.train.Saver()

    return (saver.save(sess, save_path))
