clc;    close all;    imtool close all; clear all;

addpath Data;
load Lake;

% Determine Number of frames.
[~,trainNumber] = size(DataTrain);      %Number of frames to train
[~,numberOfFrames] = size(I);
numberOfFrames = numberOfFrames;
totalFrames = trainNumber+numberOfFrames;
m = imSize(1); n= imSize(2);

% figure;
% set(gcf, 'units','normalized','outerposition',[0 0 1 1]); % Full screen.
lastFrame = uint8(zeros([m,n]));
for frame = 1 : numberOfFrames
	% Extract the frame from the data
    for i = 1 : n
    	thisFrame(:,i) = uint8(I((i-1)*m+1:i*m,frame));
    end
    Foreground = ones(size(thisFrame),'uint8') .* 255;
    
%     % Displaying it
%     subplot(221);   imshow(thisFrame);
%     caption = sprintf('Video Frame %4d', frame);
%     title(caption);
%     drawnow;                % Refresh the window.		
    
  	% Training for n frames to get the background
  	if frame == 1
%         Foreground = zeros(size(thisFrame));
        Background = thisFrame;
    elseif rem(frame,10) == 0
%        Foreground = double(lastFrame - thisFrame); 
        Background = lastFrame;
    end
        
  	% Difference between this frame and the background.
  	differenceImage =  uint8(Background) - thisFrame;
  	% Threshold
  	thresholdLevel = graythresh(differenceImage);   % Get threshold.
    binImage = im2bw( differenceImage, thresholdLevel); 
    binImage = medfilt2(binImage,[3 3]);    % Reduce noise
       
%   	% Plot the binary image.
%   	subplot(223);  	imshow(binImage);  	title('Threshold Image');
    
    %Displaying the foreground
    if thresholdLevel > 0.15
        for i = 1 : m
        for j = 1 : n
            if binImage(i,j) == 1
                Foreground(i,j) = thisFrame (i,j);
            end
        end
        end
    end
    lastFrame = thisFrame;
    
%     subplot(222);   imshow(uint8(Background));  	title('Background');
%     subplot(224);    imshow(Foreground);    title('Foreground');
end