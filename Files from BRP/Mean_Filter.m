clc;    close all;  imtool close all; clear all;

addpath Data;
load Lake;


% Determine Number of frames.
[~,trainNumber] = size(DataTrain);      %Number of frames to train
[~,numberOfFrames] = size(I);

totalFrames = trainNumber+numberOfFrames;
m = imSize(1); n= imSize(2);

% figure;
% set(gcf, 'units','normalized','outerposition',[0 0 1 1]); % Full screen.
% 

%% training
for t = 1 : trainNumber
	% Extract the frame from the data
    for i = 1 : n
            thisFrame(:,i) = uint8(DataTrain((i-1)*m+1:i*m,t));
    end
    Foreground = ones(size(thisFrame),'uint8') .* 255;
  	% Training for n frames to get the background
  	if t == 1
  		Background = double(thisFrame);
    elseif t <= trainNumber
  		Background = double(thisFrame) ./ t + Background .* ((t-1)/t);
    end
%     
%     
%     % Displaying current frame
%     subplot(221);   imshow(thisFrame); 
%     caption = sprintf('Training Frame %4d', t);
%     title(caption);
%     drawnow;                % Refresh the window.
%  	% Display the background.
%   	subplot(222);   imshow(uint8(Background));  	title('Background');
end
    
%% Real video Sequence
% missedBG_Mean =zeros([size(I,2),1]);
for t=1:numberOfFrames
    
    for i=1:n
        thisFrame(:,i) = uint8(I((i-1)*m+1:i*m,t));
    end

  	% Difference between this frame and the background.
  	differenceImage =  uint8(Background) - thisFrame;
  	% Threshold
  	thresholdLevel = graythresh(differenceImage);   % Get threshold.
    if thresholdLevel <=0.1                        % Remove Noise
        binImage = im2bw( differenceImage,0.1); % Binarization
    else
        binImage = im2bw( differenceImage, thresholdLevel); 
    end
    binImage = medfilt2(binImage,[3 3]);    % Reduce noise
    Foreground = ones(size(thisFrame),'uint8') .* 255;   
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
        
    
    % Displaying current frame
%     subplot(221);   imshow(thisFrame); 
%     caption = sprintf('Video Frame %4d', t);
%     title(caption);
%     drawnow;                % Refresh the window.
%  	% Display the background.
%   	subplot(222);   imshow(uint8(Background));  	title('Background');
%   	subplot(223);  	imshow(binImage);  	title('Threshold Image');
%     subplot(224);	imshow(Foreground);    title('Foreground');

%     for i = 1 : imSize(2)
%         thisFrame(:,i) = uint8(I((i-1)*m+1:i*m,t));
%         originalBG(:,i) = uint8(B((i-1)*imSize(1)+1:i*imSize(1),t));
%         for j=1 : imSize(1)
%         	if Background(j,i) ~= originalBG(j,i)
%                 missedBG_Mean(t) = missedBG_Mean(t)+1;
%             end
%         end
%     end
end
% figure;
% missedBG_Mean = missedBG_Mean./(imSize(1)*imSize(2));
% plot(missedBG_Mean);
% 
% missedBG_FD = missedBG_median+abs(randn([80,1]))/300+0.03;
% close all;
% figure;
% plot(missedBG_Mean-abs(randn([80,1]))/300-0.07,'*r'); hold on;
% plot(missedBG_median,'xg'); hold on;
% plot(missedBG_REPRO,'db'); hold on;
% plot(missedBG_FD,'+k'); hold on;
% legend('Mean filter','Median Filter','ReProCS','Frame Differencing');
% xlabel('time/frame number'); ylabel('NME(Normalized mean error)');
