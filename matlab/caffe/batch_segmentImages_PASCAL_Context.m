% Demo code of how to segment image using a caffe model
% You can run batch_evaluateSegments_PASCAL_Context afterwards

% change the root directory to yours
% The structure for the root directory is:
%   -- val.txt ==> contains the image names
%   -- SemanticImages ==> contains the ground truth label
%   -- Images ==> contains the original image
root_dir = '/playpen1/data/PASCAL-Context';
test_set = 'val';
fileList = sprintf('%s/%s.txt', root_dir, test_set);
fid = fopen(fileList);
C = textscan(fid, '%s');
names = C{1};
clear C
fclose(fid);

% change the model_def_file and model_file to run your own model
model_def_file = '../../models/VGGNet/PASCAL-Context/fcn-32s-pascalcontext-deploy.prototxt';
model_file = '../../models/VGGNet/PASCAL-Context/fcn-32s-pascalcontext.caffemodel';

use_gpu = 1;

matcaffe_init(use_gpu, model_def_file, model_file);

nd = length(names);
for d = 1:nd
    img_file = sprintf('%s/Images/%s.jpg', root_dir, names{d});
    img = imread(img_file);
    seg_file = sprintf('%s/results/Segmentation/comp5_%s_cls_2k/%s.png', root_dir, test_set, names{d});
    seg_dir = fileparts(seg_file);
    if ~exist(seg_dir, 'dir')
        mkdir(seg_dir);
    end
    seg = wl_segmentImage(img);
    imwrite(seg, seg_file);
end
