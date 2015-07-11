clear all; close all; clc;

%% startup
startup;
config.imageset = 'test';
config.cmap= './voc_gt_cmap.mat';
config.gpuNum = 0;
config.Path.CNN.caffe_root = './caffe';
config.save_root = './results';


%% cache FCN-8s results
config.write_file = 0;
config.Path.CNN.script_path = './FCN';
config.Path.CNN.model_data = [config.Path.CNN.script_path '/fcn-8s-pascal.caffemodel'];
config.Path.CNN.model_proto = [config.Path.CNN.script_path '/fcn-8s-pascal-deploy.prototxt'];
config.im_sz = 500;

generate_fcn_results(config);

