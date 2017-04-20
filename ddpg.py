import numpy as np
import tensorflow as tf
import tflearn
from replay_buffer import ReplayBuffer
from mahimahiInterface import *





class Network(object):

	def __init__(self, sess, input_dim, output_dim, learning_rate):
		self.sess = sess
		self.in_dim = input_dim
		self.out_dim = output_dim
		self.learning_rate = learning_rate

		# Create the critic network
		self.inputs, self.y_, self.y = self.create_network()

		self.network_params = tf.trainable_variables()
		# print self.network_params[3]



		# Define loss and optimization Op
		self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.y_, logits=self.y))
					# +  ( 1.0 / 50.0 * tf.nn.l2_loss(self.network_params[0])
					# 	 + 1.0 / 100.0 * tf.nn.l2_loss(self.network_params[2])
					# 	 + 1.0 / 80.0 * tf.nn.l2_loss(self.network_params[4]))
		# self.loss = tf.reduce_mean(-tf.reduce_sum(self.y_ * tf.log(self.y), reduction_indices=[1])) + 0.01 * (tf.nn.l2_loss(self.network_params[0])+tf.nn.l2_loss(self.network_params[2]))
		# self.optimize = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.loss)

		'''
  		decayed_learning_rate = learning_rate / (1 + decay_rate * t)
		'''
		global_step = tf.Variable(0, trainable=False)

		self.decayed_learning_rate = tf.train.inverse_time_decay(self.learning_rate, global_step=global_step, decay_steps=500, decay_rate=0.5, name='learningrate')

		self.optimize = tf.train.AdamOptimizer(self.decayed_learning_rate).minimize(self.loss, global_step=global_step)
		# self.optimize = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.loss)

		self.correct_prediction = tf.equal(tf.argmax(self.y, 1), tf.argmax(self.y_, 1))
		self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))

		tf.scalar_summary("cost", self.loss)
		tf.scalar_summary("accuracy", self.accuracy)
		tf.scalar_summary("learning_rate", self.decayed_learning_rate)

		self.summary_op = tf.merge_all_summaries()

	def create_network(self):
		with tf.name_scope('Data'):
			inputs = tflearn.input_data(shape=[None, self.in_dim], name='input')
			y_ = tf.placeholder(tf.float32, [None, self.out_dim],name = 'desired_output')

		with tf.name_scope('Weights'):
			W1 = tf.Variable(tf.truncated_normal([self.in_dim,100], mean=0.0,stddev=1.0/np.sqrt(self.in_dim)),name='l1_W')
			W2 = tf.Variable(tf.truncated_normal([100, 100], mean=0.0, stddev=1.0 / np.sqrt(100)),name='l2_W')
			W3 = tf.Variable(tf.truncated_normal([100, 50], mean=0.0, stddev=1.0 / np.sqrt(100)),name='l3_W')
			W4 = tf.Variable(tf.truncated_normal([50, self.out_dim], mean=0.0, stddev=1.0 / np.sqrt(50)),name='l4_W')
			tf.histogram_summary("W1_hist", W1)
			tf.histogram_summary("W2_hist", W2)
			tf.histogram_summary("W3_hist", W3)
			tf.histogram_summary("W4_hist", W4)


		with tf.name_scope('Biases'):
			b1 = tf.Variable(tf.zeros([100]),name='b1')
			b2 = tf.Variable(tf.zeros([100]),name='b2')
			b3 = tf.Variable(tf.zeros([50]),name='b3')
			b4 = tf.Variable(tf.zeros([self.out_dim]),name='b4')
			tf.histogram_summary("b1_hist", b1)
			tf.histogram_summary("b2_hist", b2)
			tf.histogram_summary("b3_hist", b3)
			tf.histogram_summary("b4_hist", b4)


		with tf.name_scope('Activations'):
			y1 = tf.nn.relu(tf.matmul(inputs, W1) + b1 , name='a1_relu')
			y2 = tf.nn.relu(tf.matmul(y1, W2) + b2, name='a2_relu')
			y3 = tf.nn.relu(tf.matmul(y2, W3) + b3, name='a3_relu')
			y = tf.nn.softmax(tf.matmul(y3, W4) + b4, name = 'a4_softmax')
			tf.histogram_summary("a1_hist", y1)
			tf.histogram_summary("a2_hist", y2)
			tf.histogram_summary("a3_hist", y3)
			tf.histogram_summary("a4_hist", y)


		return inputs, y_, y

	def get_w(self):
		return self.sess.run(self.network_params)

	def train(self, inputs, y_):
		return self.sess.run([self.loss, self.optimize, self.summary_op], feed_dict={
			self.inputs: inputs,
			self.y_: y_
		})

	def predict(self, inputs):
		return self.sess.run(self.y, feed_dict={
			self.inputs: inputs
		})

	def acc_eval(self, inputs, y_):
		return self.sess.run(self.accuracy, feed_dict={
			self.inputs: inputs,
			self.y_: y_
		})
    #
	# def variable_summaries(self, var):
	# 	"""Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
	# 	with tf.name_scope('summaries'):
	# 		# mean = tf.reduce_mean(var)
	# 		# tf.summary.scalar('mean', mean)
	# 		# with tf.name_scope('stddev'):
	# 		#   stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
	# 		# tf.summary.scalar('stddev', stddev)
	# 		# tf.summary.scalar('max', tf.reduce_max(var))
	# 		# tf.summary.scalar('min', tf.reduce_min(var))
	# 		tf.summary.histogram('histogram', var)
    #
	# def nn_layer(self, input_tensor, input_dim, output_dim, layer_name, act=tf.nn.relu):
	# 	"""
     #    It does a matrix multiply, bias add, and then uses relu to nonlinearize.
     #    It also sets up name scoping so that the resultant graph is easy to read,
     #    and adds a number of summary ops.
     #    """
	# 	# Adding a name scope ensures logical grouping of the layers in the graph.
	# 	with tf.name_scope(layer_name):
	# 		# This Variable will hold the state of the weights for the layer
	# 		with tf.name_scope('weights'):
	# 			weights = tf.Variable(tf.zeros([input_dim, output_dim]))
	# 			variable_summaries(weights)
	# 		with tf.name_scope('biases'):
	# 			biases = tf.Variable(tf.zeros([output_dim]))
	# 			variable_summaries(biases)
	# 		with tf.name_scope('Wx_plus_b'):
	# 			preactivate = tf.matmul(input_tensor, weights) + biases
	# 			tf.summary.histogram('pre_activations', preactivate)
	# 		activations = act(preactivate, name='activation')
	# 		tf.summary.histogram('activations', activations)
	# 		return activations

# def action_gradients(self, inputs, actions):
	# 	return self.sess.run(self.action_grads, feed_dict={
	# 		self.inputs: inputs,
	# 		self.action: actions
	# 	})

# ===========================
#   Tensorflow Summary Ops
# ===========================
def build_summaries(): 
	episode_reward = tf.Variable(0.)
	tf.scalar_summary("Reward", episode_reward)
	td_loss = tf.Variable(0.)
	tf.scalar_summary("TD", td_loss)

	summary_vars = [episode_reward, td_loss]
	summary_ops = tf.merge_all_summaries()

	return summary_ops, summary_vars

