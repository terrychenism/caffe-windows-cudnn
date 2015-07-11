function generate_fcn_results(config)

fprintf('start caching FCNs restuls and score\n');

%% initialization
load(config.cmap);

%% initialize caffe
fprintf('initializing caffe..\n');
if caffe('is_initialized')
    caffe('reset')
end
caffe('init', config.Path.CNN.model_proto, config.Path.CNN.model_data,'test')
caffe('set_device', config.gpuNum);
caffe('set_mode_gpu');
fprintf('done\n');

%% initialize paths
save_res_dir = [config.save_root, '/FCNs/results'];
save_res_path = [save_res_dir, '/%s.png'];
save_score_dir = [config.save_root, '/FCN8s/scores'];
save_score_path = [save_score_dir, '/%s.mat'];

%% create directory
if config.write_file
    if ~exist(save_res_dir), mkdir(save_res_dir), end
    if ~exist(save_score_dir), mkdir(save_score_dir), end
end

tic;

% read image
I = imread('000001.jpg');
input_data = preprocess_image(I, config.im_sz); 
cnn_output = caffe('forward', input_data);

result = cnn_output{1};

tr_result = permute(result,[2,1,3]);
score = tr_result(1:size(I,1),1:size(I,2),:);

[~, result_seg] = max(score,[], 3);   
result_seg = uint8(result_seg-1);

if config.write_file
    imwrite(result_seg, cmap, sprintf(save_res_path, ids{i}));
    save(sprintf(save_score_path, ids{i}), 'score');
else
    subplot(1,2,1);
    imshow(I);
    subplot(1,2,2);
    result_seg_im = reshape(cmap(int32(result_seg)+1,:),[size(result_seg,1),size(result_seg,2),3]);
    imshow(result_seg_im);    
end
fprintf(' done [%f]\n', toc);


%% function end
end







