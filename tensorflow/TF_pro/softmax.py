# -*- coding: utf-8 -*-
"""
Created on Sun Dec 15 22:11:30 2019

@author: Administrator
"""

from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

mnist = input_data.read_data_sets('MNIST/', one_hot=True)
print(mnist.train.images.shape, mnist.train.labels.shape)
print(mnist.test.images.shape, mnist.test.labels.shape)
print(mnist.validation.images.shape, mnist.validation.labels.shape)

#定义session
sess = tf.InteractiveSession()

#定义样本空间
x = tf.placeholder(tf.float32, [None, 784])

#定义参数，并初始化
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

#定义model：x经过加权求和后做softmax非线性化，得到类别概率[正向传播]
y = tf.nn.softmax(tf.matmul(x,W)+b)

#给真实标签申请空间
y_ = tf.placeholder(tf.float32, [None, 10])

#定义交叉熵为损失函数，定义如何来算误差
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_*tf.log(y), reduction_indices=[1]))

#定义优化方式，BP，怎么求参数 【用梯度下降法反向传播求参数，目标最小化损失函数cross_entropy】
train_step = tf.train.GradientDescentOptimizer(learning_rate=0.5).minimize(cross_entropy)

#初始化参数
tf.global_variables_initializer().run()

#训练的过程,得到参数W和b
for i in range(1000):
	#每次取100条数据
	batch_xs, batch_ys = mnist.train.next_batch(100)
	train_step.run({x:batch_xs, y_:batch_ys})

correct_pred = tf.equal(tf.argmax(y,1),tf.argmax(y_,1))

accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
print(accuracy.eval({x:mnist.test.images, y_:mnist.test.labels}))











