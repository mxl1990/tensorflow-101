# -*- coding: utf-8 -*-
# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""A very simple MNIST classifier.

See extensive documentation at
http://tensorflow.org/tutorials/mnist/beginners/index.md
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys

from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf

FLAGS = None


def main(_):
	# Import data
	# 读取数据
	mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

	# Create the model
	# None表示可变维度的向量
	# 因为可能有多张图片输入
	# 每张图784维
	x = tf.placeholder(tf.float32, [None, 784])
	# placeholder表示输入（仅输入一次）
	# Variable表示变量，一直改变
	# 784 * 10的矩阵
	W = tf.Variable(tf.zeros([784, 10]))
	# 10维向量
	b = tf.Variable(tf.zeros([10]))
	# 计算公式 y = Wx + b
	# 说明只有一层的单层神经网络
	y = tf.matmul(x, W) + b

	# Define loss and optimizer
	# 这是分类结果，第一维是图片数量
	y_ = tf.placeholder(tf.float32, [None, 10])

	# The raw formulation of cross-entropy,
	#
	#   tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(tf.nn.softmax(y)),
	#                                 reduction_indices=[1]))
	#
	# can be numerically unstable.
	#
	# So here we use tf.nn.softmax_cross_entropy_with_logits on the raw
	# outputs of 'y', and then average across the batch.

	# reduce_mean计算平均值
	cross_entropy = tf.reduce_mean(
		  # softmax的交叉熵，y_与y之间的交叉熵
			tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
	# 创建了梯度下降优化算法的对象
	# 并指定最小化损失操作
	# 这个操作等同于调用compute_gradients() 和 apply_gradients()
	train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

	# 创建工作对象
	sess = tf.InteractiveSession()
	# 初始化全局变量(使用tf.Variable定义的)
	tf.global_variables_initializer().run()
	# Train
	# 迭代一千次
	for _ in range(1000):
		# 每次取100个样本进行训练
		batch_xs, batch_ys = mnist.train.next_batch(100)
		# batch_xs->x,batch_ys->y_
		sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

	# Test trained model
	# 计算预测的标签结果和正确结果的比对，统计个数
	correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
	# tf.cast转换数据类型
	# reduce_mean，求和之后求平均值
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
	# run session的时候将具体数值指定
	print(sess.run(accuracy, feed_dict={x: mnist.test.images,
																			y_: mnist.test.labels}))

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--data_dir', type=str, default='/tmp/tensorflow/mnist/input_data',
											help='Directory for storing input data')
	FLAGS, unparsed = parser.parse_known_args()
	tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
