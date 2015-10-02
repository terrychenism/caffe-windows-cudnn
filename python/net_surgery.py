import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Make sure that caffe is on the python path:
caffe_root = '../'  # this file is expected to be in {caffe_root}/examples
import sys
sys.path.insert(0, caffe_root + 'python')

import caffe

def sample_filter(param, n_stride, c_stride, h_stride, w_stride):
  out_param = param[0::n_stride, 0::c_stride, 0::h_stride, 0::w_stride]
  return out_param.flat

# Load the original network and extract the fully connected layers' parameters.
net = caffe.Net('../models/VGGNet/VGG_ILSVRC_16_layers_fc_deploy.prototxt',
    '../models/VGGNet/VGG_ILSVRC_16_layers_fc.caffemodel', caffe.TEST)
params = net.params.keys()
fc_params = {pr: (net.params[pr][0].data, net.params[pr][1].data) for pr in params}

for fc in params:
  print '{} weights are {} dimensional and biases are {} dimensional'.format(fc, fc_params[fc][0].shape, fc_params[fc][1].shape)

# Load the fully convolutional network to transplant the parameters.
net_full_conv = caffe.Net('../models/VGGNet/VGG_ILSVRC_16_layers_fc_reduced_deploy.prototxt', caffe.TEST)

params_full_conv = net.params.keys()
# conv_params = {name: (weights, biases)}
conv_params = {pr: (net_full_conv.params[pr][0].data, net_full_conv.params[pr][1].data) for pr in params_full_conv}

for conv in params_full_conv:
  print '{} weights are {} dimensional and biases are {} dimensional'.format(conv, conv_params[conv][0].shape, conv_params[conv][1].shape)

for pr, pr_conv in zip(params, params_full_conv):
  if 'fc6' == pr:
    conv_params[pr_conv][0].flat = sample_filter(fc_params[pr][0], 4, 1, 3, 3)
    conv_params[pr_conv][1][...] = fc_params[pr][1][0::4]
  elif 'fc7' == pr:
    conv_params[pr_conv][0].flat = sample_filter(fc_params[pr][0], 4, 4, 1, 1)
    conv_params[pr_conv][1][...] = fc_params[pr][1][0::4]
  elif 'fc8' == pr:
    conv_params[pr_conv][0].flat = sample_filter(fc_params[pr][0], 1, 4, 1, 1)
    conv_params[pr_conv][1][...] = fc_params[pr][1]
  else:
    conv_params[pr_conv][0].flat = fc_params[pr][0].flat  # flat unrolls the arrays
    conv_params[pr_conv][1][...] = fc_params[pr][1]
  print 'fc: {} weights are {} dimensional and biases are {} dimensional'.format(pr, fc_params[pr][0].shape, fc_params[pr][1].shape)
  print 'conv: {} weights are {} dimensional and biases are {} dimensional'.format(pr, conv_params[pr_conv][0].shape, conv_params[pr_conv][1].shape)

net_full_conv.save('../models/VGGNet/VGG_ILSVRC_16_layers_fc_reduced.caffemodel')
