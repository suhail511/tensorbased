clear all; clc; close all;
%addpath SimulationData\
load Tensor_800NoF_300Basic.mat

TrainPath = 'SABS\Train\NoForegroundDay\';
addpath(TrainPath);
nrow = 120; ncol = nrow*160/120; p=nrow*ncol;

%% Reading the GroundTruth

%GroundTruth
srcFiles = dir(strcat(TrainPath,'*.png'));
if length(srcFiles) > videolength
    srcFiles = srcFiles(1:videolength);
end
DataTrain = zeros([p videolength]);
for i = 1 : videolength
    filename = strcat(TrainPath,srcFiles(i).name);
    I1 = uint8(rgb2gray(imread(filename)));
    I1 = imresize(I1, [nrow NaN]);
    DataTrain(:,i) = reshape(I1,[p 1]);
end

%% MSE and PSNR
MSE=0;

for t = 1 : videolength
    for i = 1 : p
        MSE = MSE + (DataTrain(i,t) - round(Bg(i,t))).^2;
    end
end
MSE = MSE ./ (p*videolength);

PSNR1 = 10.*log10(255*255 / MSE)

%% Display
close all;
figure; 
TrueBg = uint8(zeros(nrow,ncol));
ThisFrame   = uint8(zeros(nrow,ncol));
Background  = uint8(zeros(120,160,videolength));
Foreground  = uint8(zeros(120,160,videolength));
set(gcf, 'units','normalized','outerposition',[0 0 1 1]); % Full screen.
for t = 1 : videolength-2
    for i = 1 : ncol
        TrueBg(:,i)    = uint8(DataTrain((i-1)*nrow+1:i*nrow,t));
        ThisFrame(:,i)      = uint8(I((i-1)*nrow+1:i*nrow,t));
        Background(:,i,t)   = uint8(Bg((i-1)*nrow+1:i*nrow,t));
        Foreground(:,i,t)   = uint8(Fg((i-1)*nrow+1:i*nrow,t));
        for j=1 : nrow
            if Foreground(j,i,t) == 0
                Foreground(j,i,t) = 255;
            end
        end
    end
    subplot(221); imshow(TrueBg); caption = sprintf('True Background %4d', t); title(caption);
    subplot(222); imshow(Background(:,:,t)); title('Old Foreground');
    subplot(223); imshow(ThisFrame); caption = sprintf('Video Frame %4d', t); title(caption);
    subplot(224); imshow(Foreground(:,:,t)); title('New Foreground');
    pause(0.01);
end