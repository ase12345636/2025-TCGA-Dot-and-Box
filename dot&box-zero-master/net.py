import numpy as np
import tensorflow as tf

import config

tf.compat.v1.disable_eager_execution()


class NN:
    def __init__(self, session):
        self.session = session

    def f_batch(self, state_batch):
        return self.session.run([p_op, v_op], feed_dict={state: state_batch, training: False})

    def train(self, state_batch, pi_batch, z_batch):
        p_loss, v_loss, _ = self.session.run([p_loss_op, v_loss_op, train_step_op], feed_dict={
            state: state_batch,
            pi: pi_batch,
            z: z_batch,
            learning_rate: config.learning_rate,
            training: True})
        return p_loss, v_loss


def init_model():
    x = single_convolutional_block(state)
    for i in range(config.residual_blocks_num):
        x = residual_block(x)
    p_op = policy_head(x)
    v_op = value_head(x)
    with tf.compat.v1.control_dependencies(tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS)):
        v_loss_op, p_loss_op, combined_loss_op = loss(pi, z, p_op, v_op)
        train_step_op = tf.compat.v1.train.MomentumOptimizer(
            learning_rate=learning_rate, momentum=config.momentum).minimize(combined_loss_op)
    return p_op, v_op, p_loss_op, v_loss_op, train_step_op


def weight_variable(shape):
    return tf.compat.v1.Variable(tf.compat.v1.truncated_normal(shape, stddev=0.1))


def bias_variable(shape):
    return tf.compat.v1.Variable(tf.compat.v1.constant(0.1, shape=shape))


def conv2d(x, kernel_size, filter_num):
    return tf.keras.layers.Conv2D(filter_num, kernel_size, (1, 1), 'same')(x)


def batch_normalization(x):
    return tf.keras.layers.BatchNormalization()(x, training=training)


def rectifier_nonlinearity(x):
    return tf.keras.layers.ReLU()(x)


def linear_layer(x, size):
    # W = weight_variable([x.shape.dims[1].value, size])
    # b = bias_variable([size])
    # return tf.compat.v1.matmul(x, W) + b
    return tf.keras.layers.Dense(size)(x)


def single_convolutional_block(x):
    x = conv2d(x, 3, 64)
    x = batch_normalization(x)
    return rectifier_nonlinearity(x)


def residual_block(x):
    original_x = x
    x = conv2d(x, 3, 64)
    x = batch_normalization(x)
    x = rectifier_nonlinearity(x)
    x = conv2d(x, 3, 64)
    x = batch_normalization(x)
    x += original_x
    return rectifier_nonlinearity(x)


def policy_head(x):
    x = conv2d(x, 1, 2)
    x = batch_normalization(x)
    x = rectifier_nonlinearity(x)
    x = tf.keras.layers.Reshape((-1, config.board_length * 2))(x)
    return linear_layer(x, config.all_moves_num)


def value_head(x):
    x = conv2d(x, 3, 3)
    x = conv2d(x, 1, 1)
    x = batch_normalization(x)
    x = rectifier_nonlinearity(x)
    x = tf.keras.layers.Reshape((-1, config.board_length))(x)
    x = linear_layer(x, 64)
    x = rectifier_nonlinearity(x)
    x = linear_layer(x, 1)
    return tf.keras.activations.tanh(x)


def loss(pi, z, p, v):
    v_loss = tf.compat.v1.reduce_mean(tf.compat.v1.square(z - v))
    p_loss = tf.compat.v1.reduce_mean(
        tf.compat.v1.nn.softmax_cross_entropy_with_logits_v2(logits=p, labels=pi))
    variables = [v for v in tf.compat.v1.trainable_variables(
    ) if 'bias' not in v.name and 'beta' not in v.name]
    l2 = tf.compat.v1.add_n([tf.compat.v1.nn.l2_loss(variable)
                            for variable in variables])
    return v_loss, p_loss, v_loss + p_loss + config.l2_weight * l2


state = tf.compat.v1.placeholder(tf.compat.v1.float32, [
                                 None, config.N_data, config.N_data, config.history_num * 2 + 1], name='state')
pi = tf.compat.v1.placeholder(
    tf.compat.v1.float32, [None, config.all_moves_num], name='pi')
z = tf.compat.v1.placeholder(tf.compat.v1.float32, [None, 1], name='z')
training = tf.compat.v1.placeholder(tf.compat.v1.bool, name='training')
learning_rate = tf.compat.v1.placeholder(
    tf.compat.v1.float32, name='learning_rate')
p_op, v_op, p_loss_op, v_loss_op, train_step_op = init_model()
