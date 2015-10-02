function perLabelStat = wl_evaluateSegment(seg_img, gt_img, num_cls, ignore_label)

perLabelStat = zeros(num_cls, 3);
labels = unique(gt_img(:));
if exist('ignore_label', 'var') && ~isempty(ignore_label)
    labels(labels==ignore_label) = [];
    ignore_mask = gt_img == ignore_label;
end
for d = 1:length(labels)
    l = labels(d);
    label_mask = gt_img == l;
    % number of pixels with predicted label and ground truth label l
    perLabelStat(l+1, 1) = sum(seg_img(label_mask)==l);
    % number of ground truth pixels with label l
    perLabelStat(l+1, 2) = sum(label_mask(:));
    % number of predicted pixels with label l
    if exist('ignore_label', 'var') && ~isempty(ignore_label)
        perLabelStat(l+1, 3) = sum(sum(seg_img(~ignore_mask)==l));
    else
        perLabelStat(l+1, 3) = sum(sum(seg_img==l));
    end
end

pred_labels = unique(seg_img(:));
remain_pred_labels = setdiff(pred_labels, labels);
for d = 1:length(remain_pred_labels)
    l = remain_pred_labels(d);
    if exist('ignore_label', 'var') && ~isempty(ignore_label)
        perLabelStat(l+1, 3) = sum(sum(seg_img(~ignore_mask)==l));
    else
        perLabelStat(l+1, 3) = sum(sum(seg_img==l));
    end
end