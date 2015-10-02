% Demo code of how to evaluate segmentation result
% You should run batch_segmentImages_PASCAL_Context beforehand

% change the root directory to yours
% The structure for the root directory is:
%   -- val.txt ==> contains the image names
%   -- SemanticImages ==> contains the ground truth label
%   -- Images ==> contains the original image
root_dir = '/playpen1/data/SiftFlow';
test_set = 'TestSet1';
fileList = sprintf('%s/%s.txt', root_dir, test_set);
fid = fopen(fileList);
C = textscan(fid, '%s');
names = C{1};
clear C
fclose(fid);

num_cls = 33;
ignore_label = [255];
perLabelStats = zeros(num_cls, 3);

nd = length(names);
tic;
for d = 1:nd
    if toc > 1
        fprintf('%d/%d\n', d, nd);
        tic;
    end
    gt_file = sprintf('%s/SemanticImages/%s.png', root_dir, names{d});
    gt_img = imread(gt_file);
    seg_file = sprintf('%s/results/%s.png', root_dir, names{d});
    if ~exist(seg_file, 'file')
        seg = wl_segmentImage(img);
        imwrite(seg, seg_file);
    end
    seg_img = imread(seg_file);
    perLabelStat = wl_evaluateSegment(seg_img, gt_img, num_cls, ignore_label);
    perLabelStats = perLabelStats + perLabelStat;
end

%perLabelStats(all(perLabelStats==0, 2), :) = [];
perLabelStats(perLabelStats(:,2)==0, :) = [];
fprintf('%.4f ', 100 * sum(perLabelStats(:,1)) / sum(perLabelStats(:,2)));
fprintf('%.4f ', 100 * mean(perLabelStats(perLabelStats(:,2)~=0,1)./perLabelStats(perLabelStats(:,2)~=0,2)));
ious = perLabelStats(:,1) ./ (perLabelStats(:,2) + perLabelStats(:,3) - perLabelStats(:,1));
fprintf('%.4f ', 100 * mean(ious));
fprintf('%.4f\n', 100 * sum(ious .* perLabelStats(:,2)) / sum(perLabelStats(:,2)));
