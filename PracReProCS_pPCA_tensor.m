clear all; clc; close all;
addpath Yall1;
TrainPath = 'SABS\Train\NoForegroundDay\';
VideoPath = 'SABS\Test\Basic\';
nrow = 120;

addpath(TrainPath);
addpath(VideoPath);

%% Reading the files

%Background
srcFiles = dir(strcat(TrainPath,'*.png'));
DataTrain = zeros([nrow*nrow*160/120 length(srcFiles)]);
for i = 1 : length(srcFiles)
    filename = strcat(TrainPath,srcFiles(i).name);
    I1 = double(rgb2gray(imread(filename)));
    I1 = imresize(I1, [nrow NaN]);
    %imshow(I); title('Image ',i);
    DataTrain(:,i) = reshape(I1,[size(I1,1)*size(I1,2) 1]);
end

%Foreground
srcFiles = dir(strcat(VideoPath,'*.png'));
if length(srcFiles) > 300
    srcFiles = srcFiles(1:300);
end
I = zeros([nrow*nrow*160/120 length(srcFiles)]);
for i = 1 : length(srcFiles)
    filename = strcat(VideoPath,srcFiles(i).name);
    I1 = double(rgb2gray(imread(filename)));
    I1 = imresize(I1, [nrow NaN]);
    %imshow(I1); caption = sprintf('Video Frame %4d', i); title(caption);
    I(:,i) = reshape(I1,[size(I1,1)*size(I1,2) 1]);
end

imSize = size(I1);
Kmin     = 3;
Kmax     = 10;
alpha    = 20;
b        = 0.95;
p = size(I,1);
q = size(I,2);


%%%% training
mu0         = mean(DataTrain,2);
numTrain    = size(DataTrain,2);
MTrain      = DataTrain-repmat(mu0,1,numTrain);
M           = I-repmat(mu0,1, size(I,2)); %subtract the mean
[U, Sig, ~] = svd(1/sqrt(numTrain)*MTrain,0);

evals       = diag(Sig).^2;
energy      = sum(evals);
cum_evals   = cumsum(evals);
ind0        = find(cum_evals < b*energy);
rhat        = min(length(ind0),round(numTrain/10));
lambda_min  = evals(rhat);
U0          = U(:, 1:rhat);

%% tensor enhancement

% frame dimension for person.mat
n_row = 120;
n_col = 160;
% reshape to 3-dimensional model
A_ten = reshape(1/sqrt(numTrain)*MTrain, n_row, n_col, numTrain);
% higher order SVD
[core_S, U_cell, SD_cell] = hosvd(A_ten, 3);

% calculation of projection matrix
T_1 = U_cell{1}*U_cell{1}';
T_2 = U_cell{2}*U_cell{2}';

U_imp = kron(T_1, T_2)*U;
U1 = U_imp(:, 1:rhat);

%% practical-ReProCS

Shat_mod    = zeros(p,q); 
Lhat_mod    = zeros(p,q); 
Nhat_mod    = cell(q,1); 
Fg          = zeros(p,q);
xcheck      = zeros(p,q);
Tpred       = [];
Ut          = U0; 

Pstar    = Ut;
k        = 0;
K        = [];
addition = 0;
cnew     = [];
t_new    = []; time_upd = []; thresh_diff = []; thresh_base = [];

for t = 1: q
  
clear opts; 
opts.tol   = 1e-3; 
opts.print = 0;
Atf.times  = @(x) Projection(Ut,x); Atf.trans = @(y) Projection(Ut,y);
yt         = Atf.times(M(:,t));

% decide noise
if t==1
        opts.delta = norm(Atf.times(MTrain(:,numTrain)));
    else
        opts.delta = norm(Atf.times(Lhat_mod(:, t-1)));
end

if t==1||t==2
    [xp,~]   = yall1(Atf, yt, opts); % xp = argmin ||x||_1 subject to ||yt - At*x||_2 <= opts.delta
    omega(t) = sqrt(M(:,t)'*M(:,t)/p);
    That     = find(abs(xp)>=omega(t));
else
        if length(Nhat_mod{t-2})==0
            thresh(t)=0;
        else
            thresh(t)=length(intersect(Nhat_mod{t-1}, Nhat_mod{t-2}))/length(Nhat_mod{t-2});
            lambda1=length(setdiff(Nhat_mod{t-2}, Nhat_mod{t-1}))/length(Nhat_mod{t-1});
        end
        
    if thresh(t)<0.5
        [xp,~]   = yall1(Atf, yt, opts); 
        omega(t) = sqrt(M(:,t)'*M(:,t)/p);
        That=find(abs(xp)>=omega(t));
    else
        weights         = ones(p,1); 
        weights(Tpred)  = lambda1;
        opts.weights    = weights(:);
        [xp,flag]       = yall1(Atf, yt, opts); 
        [xp,ind]        = sort(abs(xp),'descend');
        Tcheck          = ind(1:round(min(1.4*length(Tpred),0.6*p)));
        xcheck(Tcheck,t)= subLS(Ut, Tcheck, yt);
        omega(t)        = sqrt(M(:,t)'*M(:,t)/p);
        That            = find(abs(xcheck(:,t))>=omega(t));
    end
end

    Shat_mod(That,t)    = subLS(Ut,That,yt);     
    Lhat_mod(:,t)       = M(:,t) - Shat_mod(:,t);
    Fg(That,t)          = I(That,t);
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
        
    if addition==1&&mod(t-t_new+1,alpha)==0
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
               Ut          = [Pstar Pnewhat];   
               
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
                   Pstar    = Ut;            
               end
           end
    end
    
end
   
   

