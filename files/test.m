clc;    close all;  clear all;  imtool close all;

addpath Data;
load Lake_MC;
	
% Determine Number of frames.
[~,trainNumber] = size(B);      %Number of frames to train
[~,numberOfFrames] = size(B);
totalFrames = trainNumber+numberOfFrames;
m = 72; n= 90;

figure;
set(gcf, 'units','normalized','outerposition',[0 0 1 1]); % Full screen.
for t=1:trainNumber
    for i = 1 : n
        thisFrame(:,i) = uint8(M((i-1)*m+1:i*m,t));         
    end
    imshow(thisFrame); caption = sprintf('Video Frame %4d', t); title(caption);
    pause(0.1)
end