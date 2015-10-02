function seg = wl_segmentImage(im)

mean_pix = [104.00698793, 116.66876762, 122.67891434];
input_data = {prepare_image(im, mean_pix)};

tic;
scores = caffe('forward', input_data);
toc;
scores = permute(scores{1}, [2 1 3]);
[~, seg] = max(scores, [], 3);
seg = uint8(seg - 1);


function images = prepare_image(im, mean_pix)
% resize to fixed input size
im = single(im);

% RGB -> BGR
im = im(:, :, [3 2 1]);

% change column-wise to row-wise for c++
images(:,:,:,1) = permute(im, [2 1 3]);

% mean BGR pixel subtraction
for c = 1:3
    images(:, :, c, :) = images(:, :, c, :) - mean_pix(c);
end