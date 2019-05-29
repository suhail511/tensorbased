clc;    
close all;  
clear all; 
imtool close all; 


movieFileName = 'Room.mp4';
trainNumber = 15; %Number of frames to train
vidObj = VideoReader(movieFileName);  

% Determine Number of frames.
numberOfFrames = vidObj.NumberOfFrames;
 
% figure;
% set(gcf, 'units','normalized','outerposition',[0 0 1 1]); % Full screen.

for frame = 1 : 100
    % Extract the frame from the video
    thisFrame = read(vidObj, frame);
    thisFrame = rgb2gray(thisFrame);
    thisFrame = imresize(thisFrame, [NaN 500]) ;
    
    %Initialization
    if frame == 1
        Mtrain = thisFrame;
    elseif frame < trainNumber
        Mtrain = [Mtrain, thisFrame];       % The Concatinated training Frames
    elseif frame == trainNumber
        Mtrain = im2double(Mtrain);         
        [P , Sig , ~ ] = svd(Mtrain, 0);    % Economy size SVD
        T0 = find(diag(Sig>0.1));           % 0.1 threshold for sigma diagonals
        P0 = P(:,T0);
        r = rank(P0);
        Sig0 = Sig(T0,T0); 
    end
end