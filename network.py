#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" 
in 
- conv 5х5 - pool 3х3 
- conv 4х4 - pool 3х3  
- conv 3х3 - pool 2х2  
- reshape - 1024 - dense -dense - mse

Validation:
train: 296.32 - 194.00
train: 326.57 - 356.00
train: 378.93 - 156.00
valid: 75.48 - 76.00
valid: 239.89 - 51.00
iteration 420: train_acc=0.2516, valid_acc=0.2653

"""

# export CUDA_VISIBLE_DEVICES=1

from __future__ import absolute_import,  division, print_function
import tensorflow as tf
import tensorflow_hub as hub
import sys
import math
import numpy as np
np.set_printoptions(precision=4, suppress=True)

from layers import *


def conv_network_1(x_image):

	# conv layers
	p1 = convPoolLayer(x_image, kernel=(5,5), pool_size=3, num_in=1, num_out=16, 
		func=tf.nn.relu, name='1') # 180 x 180
	p2 = convPoolLayer(p1, kernel=(5,5), pool_size=3, num_in=16, num_out=16, 
		func=tf.nn.relu, name='2')  # 60 x 60 
	p3 = convPoolLayer(p2, kernel=(4,4), pool_size=3, num_in=16, num_out=32, 
		func=tf.nn.relu, name='3')   # 20 x 20 
	p4 = convPoolLayer(p3, kernel=(3,3), pool_size=2, num_in=32, num_out=32, 
		func=tf.nn.relu, name='4')   # 10 x 10 
	p5 = convPoolLayer(p4, kernel=(3,3), pool_size=2, num_in=32, num_out=64, 
		func=tf.nn.relu, name='5')   # 5 x 5

	# fully-connected layers
	fullconn_input_size = 5*5*64
	p_flat = tf.reshape(p5, [-1, fullconn_input_size])

	f1 = fullyConnectedLayer(p_flat, input_size=fullconn_input_size, num_neurons=1024, 
		func=tf.nn.relu, name='F1')

	drop1 = tf.layers.dropout(inputs=f1, rate=0.4)	
	f2 = fullyConnectedLayer(drop1, input_size=1024, num_neurons=256, 
		func=tf.nn.relu, name='F2')
	
	drop2 = tf.layers.dropout(inputs=f2, rate=0.4)	
	f3 = fullyConnectedLayer(drop2, input_size=256, num_neurons=1, 
		func=None, name='F3')	 # it doesn't work with sigmoid or relu

	return f3



def inception_resnet_1(x_image):

	module = hub.Module("https://tfhub.dev/google/imagenet/inception_resnet_v2/feature_vector/1")
	bottleneck_tensor_size = 1536
	bottleneck_tensor = module(x_image)  # Features with shape [batch_size, num_features]
	FCL_input = bottleneck_tensor

	f1 = fullyConnectedLayer(
		FCL_input, input_size=bottleneck_tensor_size, num_neurons=1, 
		func=tf.sigmoid, name='F1')
	
	return f1

def inception_resnet(x_image):


	"""
	#module = hub.Module("https://tfhub.dev/google/imagenet/resnet_v2_50/classification/1")		
	module = hub.Module("https://tfhub.dev/google/imagenet/resnet_v2_152/classification/1")	
	assert height, width == hub.get_expected_image_size(module)
	bottleneck_tensor = module(resized_input_tensor)  # Features with shape [batch_size, num_features]
	"""

	module = hub.Module("https://tfhub.dev/google/imagenet/inception_resnet_v2/feature_vector/1")
	#height, width, color =  299, 299, 3
	bottleneck_tensor_size = 1536

	bottleneck_tensor = module(x_image)  # Features with shape [batch_size, num_features]
	
	print('bottleneck_tensor:', bottleneck_tensor)


	"""
	bottleneck_input = tf.placeholder_with_default(  # A placeholder op that passes through input when its output is not fed.
		bottleneck_tensor,
		shape=[None, bottleneck_tensor_size],
		name='BottleneckInputPlaceholder')
	print('bottleneck_input:', bottleneck_input)
	"""

	FCL_input = bottleneck_tensor

	f1 = fullyConnectedLayer(
		FCL_input, input_size=bottleneck_tensor_size, num_neurons=512, 
		func=tf.nn.relu, name='F1')
	
	drop1 = tf.layers.dropout(inputs=f1, rate=0.4)	
	
	f2 = fullyConnectedLayer(drop1, input_size=512, num_neurons=1, 
		func=tf.sigmoid, name='F2')

	return f2