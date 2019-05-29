clc;    close all;  clear all;  imtool close all;

addpath Data;
addpath Yall1
load Person;

% Determine Number of frames.
mu0 = mean(DataTrain,2);
[~,trainNumber] = size(DataTrain);      %Number of frames to train
[p,numberOfFrames] = size(I);
totalFrames = trainNumber + numberOfFrames;
m = imSize(1); n= imSize(2);            %Number of rows/columns in an Image

[P0, Sigma, ~] = svd(DataTrain - repmat(mu0,1,trainNumber),0);
rank0 = find(diag(Sigma)>5);  % Ignoring the Eigen values < 0.1
P0 = P0(:,rank0);
Sig0 = Sigma(rank0,rank0);

global tau D d sig_add sig_del % parameters for recursive PCA
tau=20; 
sig_del = 1; sig_add = 1; % thresholds used to update the PC matrix (Ut)
D=[]; d=0;

Ohat_CS = zeros(p,numberOfFrames); Shat_CS = zeros(p,numberOfFrames); Lhat_CS = zeros(p,numberOfFrames);
clear opts; opts.tol = 5e-3; opts.print = 0;
gamma = 5; 
clear opts; opts.tol = 5e-3; opts.print = 0;

Pt = P0; Sigt = Sig0;
for t=1:numberOfFrames
    
    At.times = @(x) Projection(Pt,x); At.trans = @(y) Projection(Pt,y);  
    yt = At.times(I(:,t));    
    if t==1
        opts.delta=0.05;
    else
        opts.delta = norm(At.times(Lhat_CS(:,t-1)-mu0),2);
    end
    % xp = argmin ||x||_1 subject to ||yt - At*x||_2 <=opts.delta
    [xp,Out] = yall1(At, yt, opts); 
    % support thresholding and least square estimation
    That = find(abs(xp)>=gamma);        
    Shat_CS(That,t) = subLS(Pt,That,yt);
    % estimate Lt and Ot
    Lhat_CS(:,t) = I(:,t) - Shat_CS(:,t);
    Ohat_CS(That,t) = I(That,t); 
    % recursive PCA
    [Pt,Sigt,~]= recursivePCA(Lhat_CS(:,t),Pt,Sigt);  
end

close all;
figure;
set(gcf, 'units','normalized','outerposition',[0 0 1 1]); % Full screen.
for t=1:numberOfFrames
    for i = 1 : n
        thisFrame(:,i) = uint8(I((i-1)*m+1:i*m,t));         
        Background(:,i) = uint8(Lhat_CS((i-1)*m+1:i*m,t));
        Foreground(:,i) = uint8(Shat_CS((i-1)*m+1:i*m,t));
    end
    subplot(221); imshow(thisFrame); caption = sprintf('Video Frame %4d', t); title(caption);
    subplot(222); imshow(uint8(Background)); title('Background');
    subplot(224); imshow(Foreground); title('Foreground');
    pause(0.2);
end