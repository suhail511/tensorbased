clear all; clc; close all;
addpath SimulationData\
load ReproCS_800NoF_300Basic.mat

GTPath = 'SABS\GT\';
addpath(GTPath);
nrow = 120; ncol = nrow*160/120; p=nrow*ncol;

%% Reading the GroundTruth

%GroundTruth
srcFiles = dir(strcat(GTPath,'*.png'));
if length(srcFiles) > videolength
    srcFiles(1:videolength) = srcFiles(801:800+videolength);
end
GT = zeros([p videolength]);
for i = 1 : videolength
    filename = strcat(GTPath,srcFiles(i).name);
    I1 = uint8(rgb2gray(imread(filename)));
    I1 = imresize(I1, [nrow NaN]);
    %imshow(I1); caption = sprintf('Video Frame %4d', i); title(caption);
    GT(:,i) = reshape(I1,[p 1]);
end

%% Smoothing Fg
%  alpha = [0.75 ];
%  Fg = Smeasure(Fg,videolength,nrow,ncol,alpha);
%% Precision, Recall and F-measure
TP=zeros([videolength 1]); FP=zeros([videolength 1]); TN =zeros([videolength 1]); FN =zeros([videolength 1]); 
precision = zeros([videolength 1]); recall = zeros([videolength 1]); Fmeasure = zeros([videolength 1]);

for t = 1 : videolength
    for i = 1 : p
        if GT(i,t) ~= 0
            if Fg(i,t) == 0 && I(i,t) ~= 0
                FN(t)=FN(t)+1;
            else
                TP(t)=TP(t)+1;
            end
         else
            if Fg(i,t) ~= 0
                FP(t) = FP(t) +1;
            else
                TN(t) = TN(t)+1;
            end
         end
     end
     precision(t) = TP(t)/(TP(t) + FP(t));
     recall(t) = TP(t)/(TP(t) + FN(t));
     Fmeasure(t) = 2 * precision(t) * recall(t) / (precision(t) + recall(t) );
end
for t = 1: videolength
     if isnan(Fmeasure(t))
         Fmeasure(t) = 0;
     end
end
mean(Fmeasure(Fmeasure~=0))

%% Display
close all;
% Set up the movie.
writerObj = VideoWriter('out.avi'); % Name it.
writerObj.FrameRate = 25; % How many frames per second.
open(writerObj); 

figure; 
GroundTruth = uint8(zeros(nrow,ncol));
ThisFrame   = uint8(zeros(nrow,ncol));
Background  = uint8(zeros(120,160,videolength));
Foreground  = uint8(zeros(120,160,videolength));
set(gcf, 'units','normalized','outerposition',[0 0 1 1]); % Full screen.
for t = 1 : videolength-2
    for i = 1 : ncol
        GroundTruth(:,i)    = uint8(GT((i-1)*nrow+1:i*nrow,t));
        ThisFrame(:,i)      = uint8(I((i-1)*nrow+1:i*nrow,t));
        Background(:,i,t)   = uint8(Bg((i-1)*nrow+1:i*nrow,t));
        Foreground(:,i,t)   = uint8(Fg((i-1)*nrow+1:i*nrow,t));
        for j=1 : nrow
            if Foreground(j,i,t) == 0
                Foreground(j,i,t) = 255;
            end
        end
    end
    subplot(221); imshow(ThisFrame); caption = sprintf('Video Frame %4d', t); title(caption);
    subplot(222); imshow(GroundTruth); caption = sprintf('Ground Truth %4d', t); title(caption);
    subplot(223); imshow(Background(:,:,t)); title('Background')
    subplot(224); imshow(Foreground(:,:,t)); title('Foreground');
    frame = getframe(gcf); % 'gcf' can handle if you zoom in to take a movie.
    writeVideo(writerObj, frame);
    pause(0.01);
end
close(writerObj);
figure;imshow3D(Foreground);
figure; plot(Fmeasure);