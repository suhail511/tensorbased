clc;    close all;  clear all;  imtool close all;

addpath Data;
addpath Yall1;
load Person;
%load WaterSurface;

% Determine Number of frames.
mu0 = mean(DataTrain,2);
[~,trainNumber] = size(DataTrain);      %Number of frames to train
[p,numberOfFrames] = size(I);
totalFrames = trainNumber + numberOfFrames;
m = imSize(1); n= imSize(2);            %Number of rows/columns in an Image

[P0, Sigma, ~] = svd(DataTrain - repmat(mu0,1,trainNumber),0);
rank0 = find(diag(Sigma)>0.1);  % Ignoring the Eigen values < 0.1
P0 = P0(:,rank0);
Sig0 = Sigma(rank0,rank0);

global tau D d sig_add sig_del % parameters for recursive PCA
tau=5; 
sig_del = 1; sig_add = 1; % thresholds used to update the PC matrix (Ut)
D=[]; d=0;

Shat_mod = zeros(p,numberOfFrames); Lhat_mod = zeros(p,numberOfFrames); Ohat_mod = zeros(p,numberOfFrames); Nhat_mod=cell(numberOfFrames,1);
alpha_add = 5;
alpha_del = 10;
gamma = 5; 

t=1;
Pt = P0; Sigt = Sig0;
clear opts; opts.tol = 1e-3; opts.print = 0;
D=[]; d=0;
opts.delta= 0.05;
Atf.times = @(x) Projection(Pt,x); Atf.trans = @(y) Projection(Pt,y);
yt = Atf.times(I(:,t)-mu0);
[xp,~] = yall1(Atf, yt, opts); % xp = argmin ||x||_1 subject to ||yt - At*x||_2 <= opts.delta

%support thresholding and least square
That = find(abs(xp)>=gamma);  
Shat_mod(That,t) = subLS(Pt,That,yt);     
Lhat_mod(:,t) = I(:,t) - Shat_mod(:,t);

%estimate Ot 
Ohat_mod(That ,t) = I(That ,t);
Nhat_mod{t}=That;

for t=2:numberOfFrames
    Atf.times = @(x) Projection(Pt,x); Atf.trans = @(y) Projection(Pt,y);
    yt = Atf.times(I(:,t)-mu0);
    Tpred = Nhat_mod{t-1};  %predicted support (previous support estimation)
    weights= ones(p,1); weights(Tpred)=0;
    opts.weights = weights(:);
    opts.delta = norm(Atf.times(Lhat_mod(:,t-1)-mu0),2); 
    % xp = argmin ||x_{Tpred^c}||_1 subject to ||yt - At*x||_2 <= opts.delta
    [xp,flag] = yall1(Atf, yt, opts); 

    % Add-LS-Del-LS step
    That = union(Nhat_mod{t-1},find(abs(xp)>alpha_add));
    Shat_mod(That,t) = subLS(Pt,That,yt);
    Tdel = find(abs(Shat_mod(:,t))<alpha_del); That = setdiff(That,Tdel);
    Shat_mod(That,t) = subLS(Pt,That,yt); Shat_mod(Tdel,t) = 0;

    % estimate Lt and Ot
    Ohat_mod(That,t) = I(That,t);        
    Lhat_mod(:,t) = I(:,t) - Shat_mod(:,t);
    Nhat_mod{t} = That;  
    
    % recursive PCA
    [Pt,Sigt,~]=recursivePCA(Lhat_mod(:,t)-mu0,Pt,Sigt);
end

Shat_mod2 = zeros(size(Shat_mod));
for t=1:numberOfFrames
    for o = 1:m*n;
        if I(o,t) > 0.85*Lhat_mod(o,t) && I(o,t) < 1.25*Lhat_mod(o,t)
            Shat_mod2(o,t) = 0;
        else
            Shat_mod2(o,t) = 255;
        end
    end
end
close all;
figure;
set(gcf, 'units','normalized','outerposition',[0 0 1 1]); % Full screen.
for t=1:numberOfFrames
    for i = 1 : n
        thisFrame(:,i) = uint8(I((i-1)*m+1:i*m,t));         
        Background(:,i) = uint8(Lhat_mod((i-1)*m+1:i*m,t));
        Foreground(:,i) = uint8(Shat_mod2((i-1)*m+1:i*m,t));
    end
    subplot(221); imshow(thisFrame); caption = sprintf('Video Frame %4d', t); title(caption);
    subplot(222); imshow(uint8(Background)); title('Background');
    subplot(224); imshow(Foreground); title('Foreground');
    pause(0.2);
end