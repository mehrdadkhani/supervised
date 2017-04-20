import tensorflow as tf
import tflearn
import numpy as np


class Network(object):
    def __init__(self, sess, learning_rate,inputdim):
        self.sess = sess
        self.inputdim = inputdim
        self.learning_rate = learning_rate

        self.inputs, self.out = self.create_network()

        self.predicted_out = tf.placeholder(tf.float32, [None, 1])

        # Define loss and optimization Op
        self.loss = tflearn.mean_square(self.predicted_out, self.out)


        self.optimize = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.loss)

    def create_network(self):
        inputs = tflearn.input_data(shape=[None, self.inputdim])

        W1 = tf.Variable(tf.truncated_normal([self.inputdim, 10], mean=0.0, stddev=1.0 / np.sqrt(self.inputdim)))
        b1 = tf.Variable(tf.zeros([10]))
        y1 = tf.nn.relu(tf.matmul(inputs, W1) + b1)


        W2 = tf.Variable(tf.truncated_normal([10, 10], mean=0.0, stddev=1.0 / np.sqrt(10)))
        b2 = tf.Variable(tf.zeros([10]))
        y2 = tf.nn.relu(tf.matmul(y1, W2) + b2)

        W3 = tf.Variable(tf.truncated_normal([10, 1], mean=0.0, stddev=1.0 / np.sqrt(10)))
        b3 = tf.Variable(tf.constant(0.0, shape=[1]))
        out = tf.matmul(y2, W3) + b3

        return inputs, out


    def train(self, inputs, predicted_out):
        return self.sess.run([self.loss, self.optimize], feed_dict={
            self.inputs: inputs,
            self.predicted_out: predicted_out
        })

    def predict(self, inputs, action):
        return self.sess.run(self.out, feed_dict={
            self.inputs: inputs,
        })
