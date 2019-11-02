import tensorflow as tf
from tensorflow.contrib import rnn


class CRNN():
    def __init__(self, training, num_classes):
        self.__training = training
        self.__num_classes = num_classes

    def __cnn(self, input_tensor):
        # layer1
        net = tf.layers.conv2d(inputs=input_tensor,
                               filters=64,
                               kernel_size=3,
                               strides=1,
                               padding='same',
                               use_bias=False,
                               kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                               kernel_regularizer=tf.contrib.layers.l2_regularizer(0.003),
                               name='conv1')
        net = tf.layers.batch_normalization(inputs=net,
                                            training=self.__training,
                                            name='bn1')
        net = tf.nn.leaky_relu(features=net, name='relu1')
        net = tf.layers.max_pooling2d(inputs=net,
                                      pool_size=2,
                                      strides=2,
                                      padding='same',
                                      name='pool1')
        # layer2
        net = tf.layers.conv2d(inputs=net,
                               filters=128,
                               kernel_size=3,
                               strides=1,
                               padding='same',
                               use_bias=False,
                               kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                               kernel_regularizer=tf.contrib.layers.l2_regularizer(0.003),
                               name='conv2')
        net = tf.layers.batch_normalization(inputs=net,
                                            training=self.__training,
                                            name='bn2')
        net = tf.nn.leaky_relu(features=net, name='relu2')
        net = tf.layers.max_pooling2d(inputs=net,
                                      pool_size=2,
                                      strides=2,
                                      padding='same',
                                      name='pool2')
        # layer3
        net = tf.layers.conv2d(inputs=net,
                               filters=256,
                               kernel_size=3,
                               strides=1,
                               padding='same',
                               use_bias=False,
                               kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                               kernel_regularizer=tf.contrib.layers.l2_regularizer(0.003),
                               name='conv3')
        net = tf.layers.batch_normalization(inputs=net,
                                            training=self.__training,
                                            name='bn3')
        net = tf.nn.leaky_relu(features=net, name='relu3')
        net = tf.layers.max_pooling2d(inputs=net,
                                      pool_size=[2, 1],
                                      strides=[2, 1],
                                      padding='same',
                                      name='pool3')
        # layer4
        net = tf.layers.conv2d(inputs=net,
                               filters=512,
                               kernel_size=3,
                               strides=1,
                               padding='same',
                               use_bias=False,
                               kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                               kernel_regularizer=tf.contrib.layers.l2_regularizer(0.003),
                               name='conv4')
        net = tf.layers.batch_normalization(inputs=net,
                                            training=self.__training,
                                            name='bn4')
        net = tf.nn.leaky_relu(features=net, name='relu4')
        net = tf.layers.max_pooling2d(inputs=net,
                                      pool_size=[2, 1],
                                      strides=[2, 1],
                                      padding='same',
                                      name='pool4')
        # layer5
        net = tf.layers.conv2d(inputs=net,
                               filters=512,
                               kernel_size=[2, 1],
                               strides=1,
                               padding='valid',
                               use_bias=False,
                               kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                               kernel_regularizer=tf.contrib.layers.l2_regularizer(0.003),
                               name='conv5')
        net = tf.layers.batch_normalization(inputs=net,
                                            training=self.__training,
                                            name='bn5')
        net = tf.nn.leaky_relu(features=net, name='relu5')
        assert net.get_shape().as_list()[1] == 1  # H of the feature map must equal to 1
        return tf.squeeze(net, axis=1, name='cnn_out')

    def __rnn(self, input_tensor, seq_len):
        n_unit = 256
        n_layer = 2
        lstm_cells_fw = [tf.nn.rnn_cell.LSTMCell(num_units=nu) for nu in [n_unit] * n_layer]
        lstm_cells_bw = [tf.nn.rnn_cell.LSTMCell(num_units=nu) for nu in [n_unit] * n_layer]
        net, _, _ = rnn.stack_bidirectional_dynamic_rnn(cells_fw=lstm_cells_fw,
                                                        cells_bw=lstm_cells_bw,
                                                        inputs=input_tensor,
                                                        dtype=tf.float32,
                                                        sequence_length=seq_len,
                                                        scope='bilstm')
        w = tf.Variable(tf.truncated_normal([n_layer * n_unit, self.__num_classes], stddev=0.01), name='w')
        b = tf.Variable(tf.truncated_normal([self.__num_classes], stddev=0.01), name='b')
        logits = tf.matmul(tf.reshape(net, [-1, n_layer * n_unit]), w) + b
        logits = tf.reshape(logits, [input_tensor.get_shape().as_list()[0], -1, self.__num_classes])
        return tf.transpose(logits, (1, 0, 2), name='rnn_out')

    def build(self, images, seq_len):
        cnn_out = self.__cnn(images)
        rnn_out = self.__rnn(cnn_out, seq_len)
        return rnn_out
