import numpy as np
import tensorflow as tf
import os
#import scipy.io as sio
import time
import gzip
import collections
import math as m


class DQN:
	def __init__(self, session: tf.Session, input_size: int, output_size: int, name: str = "main") -> None:
		"""DQN Agent can

		1) Build network
		2) Predict Q_value given state
		3) Train parameters

		Args:
			session (tf.Session): Tensorflow session
			input_size (int): Input dimension
			output_size (int): Number of discrete actions
                        name (str, optional): TF Graph will be built under this name scope

                """
		self.session = session
		self.input_size = input_size
		self.output_size = output_size
		self.net_name = name
		self._build_network()

	def relu(x):
		layer_1 = tf.nn.relu(x)
		return layer_1

	def weight_variable(shape):
		initial = tf.truncated_normal(shape, stddev=0.1)
		return tf.Variable(initial)

	def _build_network(self, h_size=64, l_rate=1.0e-4) -> None:

		with tf.variable_scope(self.net_name):
			self.input_data = tf.placeholder(tf.float32, [None, self.input_size*3], name='input_data')
			self.phase = tf.placeholder(tf.int64)
			net = self.input_data
			flag = self.phase
			if flag ==0:
				phase = False
			else:
				phase = True


			net = tf.layers.dense(net, 512)
			net = DQN.relu(net)
			# net = tf.contrib.layers.batch_norm(net, center=True, scale=True, is_training=phase)
			net = tf.layers.dense(net, 256)
			net = DQN.relu(net)  # (?, 500, 10)
			# net = tf.contrib.layers.batch_norm(net, center=True, scale=True, is_training=phase)
			net = tf.layers.dense(net, 128)
			net = DQN.relu(net)
			# net = tf.contrib.layers.batch_norm(net, center=True, scale=True, is_training=phase)
			net = tf.layers.dense(net, 64)
			net = DQN.relu(net)
			# net = tf.contrib.layers.batch_norm(net, center=True, scale=True, is_training=phase)

			# net = DQN.conv_end(net, Wc_4, bc_4)
			net = tf.layers.dense(net, 32)
			net = DQN.relu(net)
			# net = tf.contrib.layers.batch_norm(net, center=True, scale=True, is_training=phase)

			net = tf.layers.dense(net, h_size)
			net = DQN.relu(net)
			# net = tf.contrib.layers.batch_norm(net, center=True, scale=True, is_training=phase)
			net = tf.layers.dense(net, self.output_size)

			self._Qpred = net

			self._Y = tf.placeholder(tf.float32, shape=[None, self.output_size])
			self._loss = tf.losses.mean_squared_error(self._Y, self._Qpred)

			optimizer = tf.train.AdamOptimizer(learning_rate=l_rate)
			#optimizer = tf.train.MomentumOptimizer(learning_rate=l_rate, momentum=0.9)
			self._train = optimizer.minimize(self._loss)


	def predict(self, state: np.ndarray) -> np.ndarray:

		state = np.reshape(state, [-1, self.input_size*3])
		return self.session.run(self._Qpred, feed_dict={self.input_data: state, self.phase: 0})

	def update(self, x_stack: np.ndarray, y_stack: np.ndarray) -> list:

		feed = {
			self.input_data: x_stack,
			self._Y: y_stack,
			self.phase: 1
		}
		return self.session.run([self._loss, self._train], feed)
