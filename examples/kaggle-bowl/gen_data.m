clear all; close all; clc
s = 48;
d = dir('train');

isub = [d(:).isdir];
nameFolds = {d(isub).name}';
NNtake = length(nameFolds);
features = double(zeros(30336,s*s+1) );

start2 = 0;
for m = 3:NNtake
    fprintf('%d/%d...\n', m, NNtake);
    folderName = ['train\',nameFolds{m}];
    srcFiles = getImageSet(folderName);
    Ntake = length(srcFiles);
    subfeatures = double(zeros(Ntake,s*s+1) );
    for i = 1: Ntake
        filename= char(srcFiles(i));
        XX = imread(filename);
        img_size = size(XX);
        if  size(img_size,2) == 3
            XX = rgb2gray(XX);
        end
        X = XX/255;
        thr =  mean(X(:));
        XX = im2bw(XX, thr);
        XX = imresize(XX, [s s]);
        imwrite(XX,filename);
    end
end

