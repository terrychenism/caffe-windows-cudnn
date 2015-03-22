clear all; close all; clc
s = 48;
d = dir('pos');
Ntake = length(d);
NNtake = length(d);
for i = 3: Ntake
        
        filename= char(d(i).name);
        newfile = ['new\', filename];
        filename = ['pos\', filename];
        XX = imread(filename);
        
        XX = imresize(XX, [s s]);
        imwrite(XX,newfile);
    end
