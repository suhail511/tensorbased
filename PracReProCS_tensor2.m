clear all; close all;clc;
addpath Yall1;
TrainPath = 'SABS\Train\NoForegroundDay\'; numTrain = 800;
VideoPath = 'SABS\Test\Basic\'; videolength = 300;
nrow = 120; ncol = nrow * 160/120;% frame dimension
p = nrow*ncol;

addpath(TrainPath);
addpath(VideoPath);

%% Reading the files

%Background
srcFiles = dir(strcat(TrainPath,'*.png'));
if length(srcFiles) > numTrain
    srcFiles = srcFiles(1:numTrain);
end
DataTrain       = zeros([p numTrain]);
tensorMTrain    = zeros(nrow,ncol,numTrain);
tensorDataTrain = zeros(nrow,ncol,numTrain);
tensorM         = zeros(nrow,ncol,numTrain);
for i = 1 : length(srcFiles)
    filename = strcat(TrainPath,srcFiles(i).name);
    I1 = double(rgb2gray(imread(filename)));
    I1 = imresize(I1, [nrow NaN]);
    %imshow(I); title('Image ',i);
    DataTrain(:,i) = reshape(I1,[size(I1,1)*size(I1,2) 1]);
    tensorDataTrain(:,:,i) = I1;
end

%Foreground
srcFiles = dir(strcat(VideoPath,'*.png'));
if length(srcFiles) > videolength
    srcFiles = srcFiles(1:videolength);
end
I = zeros([p videolength]);
for i = 1 : videolength
    filename = strcat(VideoPath,srcFiles(i).name);
    I1 = double(rgb2gray(imread(filename)));
    I1 = imresize(I1, [nrow NaN]);
    %imshow(I1); caption = sprintf('Video Frame %4d', i); title(caption);
    I(:,i) = reshape(I1,[p 1]);
    tensorM(:,:,i) = I1;
end

b = 0.95;
Kmin     = 3;
Kmax     = 10;
alpha    = 20;

mu0         = mean(DataTrain,2);
mu0_tensor  = mean(tensorDataTrain,3);
MTrain      = DataTrain-repmat(mu0,1,numTrain);
M           = I-repmat(mu0,1, videolength);
for t=1:numTrain
    tensorMTrain(:,:,t) = tensorDataTrain(:,:,t) - mu0_tensor;
end
for t=1:videolength
    tensorM(:,:,t)       = tensorM(:,:,t) - mu0_tensor;
end
%% SVD
[Usvd, Sig, ~] = svd(1/sqrt(numTrain)*MTrain,0);

%Keeping b% energy
evals1       = diag(Sig).^2;
energy1      = sum(evals1);
cum_evals1   = cumsum(evals1);
ind01        = find(cum_evals1 < b*energy1);
rhat1        = min(length(ind01),round(numTrain/10));
lambda_min1  = evals1(rhat1);
U1           = Usvd(:, 1:rhat1); 

%% HOSVD (Higher Order SVD)
[S, U_cell, SD_cell] = hosvd((1/sqrt(numTrain)).*tensorMTrain);

%Keeping b% energy
for unf = 1:3
    evals{unf}          = SD_cell{unf}.^2;
    energy(unf)         = sum(evals{unf});
    cum_evals{unf}      = cumsum(evals{unf});
    ind0{unf}           = find(cum_evals{unf} < b*energy(unf));
    rhat{unf}           = min(length(ind0{unf}),round(numTrain/10));
    lambda_min{unf}     = evals{unf}(rhat{unf});
    U_cell0{unf}        = U_cell{unf}(:, 1:rhat{unf});
    SD_cell0{unf}       = SD_cell{unf}(1:rhat{unf});
end
S0 = S(1:rhat{1},1:rhat{2},1:rhat{3});
U_hat = nmode_product(S0,U_cell0{1},1); U_hat = nmode_product(U_hat,U_cell0{2},2); U_hat = nmode_product(U_hat,inv(diag(SD_cell0{3})),3);

U = (unfolding(U_hat,3))';

%% Tensor based ReProCS
    Pstar    = U;
    k        = 0;
    K        = [];
    addition = 0;
    t_new    = []; thresh_diff = []; thresh_base = [];
    Shat_mod	=   zeros(p,videolength);
    Lhat_mod	=   zeros(p,videolength);
    Nhat_mod	=   cell(videolength,1);
    omega       =   zeros(1,numTrain);
    thresh      =   zeros(1,numTrain);
    Fg        	=   zeros(p,videolength);
    Bg          =   255.*ones(p,videolength);
    xcheck      =   zeros(p,videolength);
    Tpred       =   [];
    
    for t = 1: videolength
        clear opts;
        opts.tol   = 1e-3;
        opts.print = 0;
        Atf.times  = @(x) Projection(U,x); Atf.trans = @(y) Projection(U,y);
        yt         = Atf.times(M(:,t));
        
        if t==1
            opts.delta = norm(Atf.times(reshape(tensorMTrain(:,:,end),[p 1])));
        else
            opts.delta = norm(Atf.times(Lhat_mod(:, t-1)));
        end
    
        if t==1||t==2
            [xp,~]   = yall1(Atf, yt, opts); % xp = argmin ||x||_1 subject to ||yt - At*x||_2 <= opts.delta
            omega(t) = sqrt(M(:,t)'*M(:,t)./(p).*2);
            That     = find(abs(xp)>=omega(t));
        else
            if isempty(Nhat_mod{t-2})
                thresh(t)=0;
            else
                thresh(t)=length(intersect(Nhat_mod{t-1}, Nhat_mod{t-2}))/length(Nhat_mod{t-2});
                lambda1=length(setdiff(Nhat_mod{t-2}, Nhat_mod{t-1}))/length(Nhat_mod{t-1});
            end
    
            if thresh(t)<0.5
                [xp,~]   = yall1(Atf, yt, opts); 
                omega(t) = sqrt(M(:,t)'*M(:,t)/p.*2);
                That=find(abs(xp)>=omega(t));
            else
                weights         = ones(p,1); 
                weights(Tpred)  = lambda1;
                opts.weights    = weights(:);
                [xp,~]          = yall1(Atf, yt, opts); 
                [xp,ind]        = sort(abs(xp),'descend');
                Tcheck          = ind(1:round(min(1.4*length(Tpred),0.6*p)));
                xcheck(Tcheck,t)= subLS(U, Tcheck, yt);
                omega(t)        = sqrt(M(:,t)'*M(:,t)./p.*1.3);
                That            = find(abs(xcheck(:,t))>=omega(t));
            end
        end

        Shat_mod(That,t)    = subLS(U,That,yt);     
        Lhat_mod(:,t)       = M(:,t) - Shat_mod(:,t);
        Fg(That,t)          = Shat_mod(That,t) + mu0(That);
        Nhat_mod{t}         = That;
        Tpred               = That;
        Bg(:,t)             = Lhat_mod(:,t) + mu0;
%% Projection PCA   
%         if mod(t,alpha)==0
%             Pnewhat  = [];
%             U        = Pstar;
%             D        = Lhat_mod(:,t-alpha+1:t)-Pstar*(Pstar'*Lhat_mod(:,t-alpha+1:t)); 
%             [Pnew_hat, Lambda_new,~] = svd(D./sqrt(alpha),0);
%             Lambda_new               = diag(Lambda_new).^2;
%             Lambda_new               = Lambda_new(Lambda_new>=lambda_min1);
%             th                       = round(rhat1/3);
%             if size(Lambda_new,1)> th
%                 Lambda_new=Lambda_new(1:th);
%             end
%             if numel(Lambda_new)~=0
%                 cnew_hat    = numel(Lambda_new);
%                 Pnewhat_old = Pnewhat;
%                 Pnewhat     = Pnew_hat(:,1:cnew_hat);
%                 Pstar      = U;
%                 U          = [U Pnewhat];
%             end
%         end
 end
%% Smoothing Fg
% alpha = [1.4 1.0];
% Fg = Smeasure(Fg,videolength,nrow,ncol,alpha);
%% Display

close all;
figure;
set(gcf, 'units','normalized','outerposition',[0 0 1 1]); % Full screen.
thisFrame   = uint8(zeros(120,160));
Background 	= uint8(zeros(120,160));
Foreground  = uint8(zeros(120,160,videolength));
for t=1:videolength
    for i = 1 : ncol
        thisFrame(:,i)      = uint8(I((i-1)*nrow+1:i*nrow,t));         
        Background(:,i)     = uint8(Bg((i-1)*nrow+1:i*nrow,t));
        Foreground(:,i,t)   = uint8(Fg((i-1)*nrow+1:i*nrow,t));
        for j=1 : nrow
            if Foreground(j,i,t) == 0 
                Foreground(j,i,t) = 255;
            end
        end
    end
    subplot(221); imshow(thisFrame); caption = sprintf('Video Frame %4d', t); title(caption);
    subplot(222); imshow(Background); title('Background');
    subplot(224); imshow(Foreground(:,:,t)); title('Foreground');
    pause(0.01);
end
figure;imshow3D(Foreground);

save('Tensor_800NoF_300Basic.mat' , 'Fg', 'Bg','I' ,'videolength');