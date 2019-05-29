%% The video files are unavailable
% Comments are minimum since I was working individually

clear all; clc;close all;clc;
addpath Yall1;
TrainPath = 'SABS\Train\NoForegroundDay\'; numTrain = 200;
VideoPath = 'SABS\Test\Basic\'; videolength = 300;
nrow = 120; ncol = nrow * 160/120;% frame dimension

addpath(TrainPath);
addpath(VideoPath);

%% Reading the files

%Background
srcFiles = dir(strcat(TrainPath,'*.png'));
if length(srcFiles) > numTrain
    srcFiles = srcFiles(1:numTrain);
end
DataTrain       = zeros([nrow*ncol numTrain]);
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
I = zeros([nrow*ncol videolength]);
for i = 1 : videolength
    filename = strcat(VideoPath,srcFiles(i).name);
    I1 = double(rgb2gray(imread(filename)));
    I1 = imresize(I1, [nrow NaN]);
    %imshow(I1); caption = sprintf('Video Frame %4d', i); title(caption);
    I(:,i) = reshape(I1,[nrow*ncol 1]);
    tensorM(:,:,i) = I1;
end

b = 0.95;
Kmin     = 3;
Kmax     = 10;
alpha    = 20;

p = size(I,1);
q = size(I,2);
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

%% HOSVD (Higher Order SVD)
[S, U_cell, SD_cell] = hosvd((1/sqrt(numTrain)).*tensorMTrain);

U0 = kron(U_cell{1},U_cell{2})';
SD0 = sort(kron(SD_cell{1},SD_cell{2}) ,'descend');

evals           = SD0.^2;
energy          = sum(evals);
cum_evals       = cumsum(evals);
ind0            = find(cum_evals < b*energy);
rhat            = min(length(ind0),round(numTrain/10));
lambda_min      = evals(rhat);
U               = U0(:, 1:rhat);

% S0 = S(1:rhat{1},1:rhat{2},1:rhat{3});
% lowRankTrain = nmode_product(S0,U_cell0{1},1); lowRankTrain = nmode_product(lowRankTrain,U_cell0{2},2); lowRankTrain = nmode_product(lowRankTrain,U_cell0{3},3);

%% Tensor based ReProCS
    Pstar    = U;
    k        = 0;
    K        = [];
    addition = 0;
    cnew     = [];
    t_new    = []; time_upd = []; thresh_diff = []; thresh_base = [];
    Shat_mod	=   zeros(p,q);
    Lhat_mod	=   zeros(p,q);
    Nhat_mod	=   cell(q,1);
    omega       =   zeros(1,numTrain);
    thresh      =   zeros(1,numTrain);
    Fg        	=   255.*ones(p,q);
    Bg          =   255.*ones(p,q);
    xcheck      =   zeros(p,q);
    Tpred       =   [];
    
    for t = 1: q
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
            omega(t) = sqrt(M(:,t)'*M(:,t)./p.*2);
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
    if addition==0    %&& norm( (Lhat(:,t-alpha+1:t) - Phat*(Phat'*Lhat(:,t-alpha+1:t)))./sqrt(alpha) )>thresh
        addition = 1;
        t_new    = t;
        Pnewhat  = [];
        k        = 0;
    end
        
    if addition==1 && mod(t-t_new+1,alpha)==0
        time_upd = [time_upd,t];           
        D        = Lhat_mod(:,t-alpha+1:t)-Pstar*(Pstar'*Lhat_mod(:,t-alpha+1:t)); 
       
        [Pnew_hat, Lambda_new,~] = svd(D./sqrt(alpha),0);
        Lambda_new               = diag(Lambda_new).^2;
        Lambda_new               = Lambda_new(Lambda_new>=lambda_min);
        th                       = round(rhat/3);
        if size(Lambda_new,1)> th
            Lambda_new=Lambda_new(1:th);
        end
           if numel(Lambda_new)==0
               addition  = 0; 
               cnew      = [cnew 0];
           else              
               cnew_hat    = numel(Lambda_new);
               Pnewhat_old = Pnewhat;
               Pnewhat     = Pnew_hat(:,1:cnew_hat); cnew = [cnew cnew_hat];%size(Pnewhat,2)];
               U          = [Pstar Pnewhat];   
               
               k=k+1;
               
               if k==1 
                   temp        =(Pnewhat*(Pnewhat'*Lhat_mod(:,t-alpha+1:t)));
                   thresh_base = [thresh_base norm(temp./sqrt(alpha))];
                   thresh_diff = [thresh_diff norm(temp./sqrt(alpha))];                 
               else
                   temp        =(Pnewhat*(Pnewhat'*Lhat_mod(:,t-alpha+1:t)));
                   thresh_base = [thresh_base norm(temp./sqrt(alpha))];
                   
                   temp        = (Pnewhat*(Pnewhat'*Lhat_mod(:,t-alpha+1:t)) - Pnewhat_old*(Pnewhat_old'*Lhat_mod(:,t-alpha+1:t)));
                   thresh_diff = [thresh_diff norm(temp./sqrt(alpha))];  
               end
               
               flag = 0;
               if k >= Kmin
                   numK = 3;
                   flag = thresh_diff(end)/thresh_base(end-1)<0.01;
                   for ik = 1:numK-1
                       flag = flag && thresh_diff(end-ik)/thresh_base(end-ik-1)<0.01;
                   end
               end
               
               if  k==Kmax|| (k>=Kmin && flag==1)                  
                   addition =0;
                   K        = [K k];
                   Pstar    = U;            
               end
            end
        end
    end

%% Display
   
close all;
figure;
set(gcf, 'units','normalized','outerposition',[0 0 1 1]); % Full screen.
thisFrame = uint8(zeros(nrow,ncol));
Background = uint8(zeros(nrow,ncol));
Foreground  = uint8(zeros(120,160,videolength));
for t=1:q
    for i = 1 : ncol
        thisFrame(:,i)      = uint8(I((i-1)*nrow+1:i*nrow,t));         
        Background(:,i)     = uint8(Bg((i-1)*nrow+1:i*nrow,t));
        Foreground(:,i,t)   = uint8(Fg((i-1)*nrow+1:i*nrow,t));
    end
    subplot(221); imshow(thisFrame); caption = sprintf('Video Frame %4d', t); title(caption);
    subplot(222); imshow(Background); title('Background');
    subplot(224); imshow(Foreground(:,:,t)); title('Foreground');
    pause(0.02);
end
figure;imshow3D(Foreground);

%Save simulation file to display later
save('Tensor1_200NoF_300Basic.mat' , 'Shat_mod', 'Lhat_mod','I', 'Nhat_mod' , 'mu0','videolength','That');