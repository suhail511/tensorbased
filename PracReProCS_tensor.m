clear all; clc;close all;clc;
addpath Yall1;
addpath Data;
addpath('./Tensor_Toolbox');

load Person.mat
b = 0.95;
% frame dimension
nrow = imSize(1);
ncol = imSize(2);

p = size(I,1);
q = size(I,2);
numTrain    = size(DataTrain,2);
mu0         = mean(DataTrain,2);
MTrain      = DataTrain-repmat(mu0,1,numTrain);
M           = I-repmat(mu0,1, size(I,2));
%% "Tensorifying" the Data

tensorMTrain    = zeros(nrow,ncol,numTrain);
tensorM         = zeros(nrow,ncol,numTrain);
for t = 1:numTrain
    for i = 1:ncol
        tensorMTrain(1:nrow,i,t) = MTrain((i-1)*nrow+1:i*nrow,t);
    end
end

for t = 1:q
    for i = 1:ncol
        tensorM((1:nrow),i,t) = M((i-1)*nrow+1:i*nrow,t);
    end
end

%% HOSVD (Higher Order SVD)
[S, U_cell, SD_cell] = hosvd((1/sqrt(numTrain)).*tensorMTrain);

%Keeping b% energy
% for unf = 1:3
%     evals{unf}          = SD_cell{unf}.^2;
%     energy(unf)         = sum(evals{unf});
%     cum_evals{unf}      = cumsum(evals{unf});
%     ind0{unf}           = find(cum_evals{unf} < b*energy(unf));
%     rhat{unf}           = min(length(ind0{unf}),round(numTrain/10));
%     lambda_min{unf}     = evals{unf}(rhat{unf});
%     U_cell0{unf}        = U_cell{unf}(:, 1:rhat{unf});
%     SD_cell0{unf}       = SD_cell{unf}(1:rhat{unf});
%     
%     % Projection matrices
% %     U{unf}              = U_cell0{unf}*U_cell0{unf}';
% end
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

%     U   =  kron(U_cell0{1},U_cell0{2});
    
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
    end

%% Display
   
close all;
figure;
set(gcf, 'units','normalized','outerposition',[0 0 1 1]); % Full screen.
% thisFrame  = zeros(120,160);
% Background = zeros(120,160);
% Foreground = zeros(120,160);
for t=1:q
    for i = 1 : imSize(2)
        thisFrame(:,i)  = uint8(I((i-1)*imSize(1)+1:i*imSize(1),t));         
        Background(:,i) = uint8(Bg((i-1)*imSize(1)+1:i*imSize(1),t));
        Foreground(:,i,t) = uint8(Fg((i-1)*imSize(1)+1:i*imSize(1),t));
    end
    subplot(221); imshow(thisFrame); caption = sprintf('Video Frame %4d', t); title(caption);
    subplot(222); imshow(Background); title('Background');
    subplot(224); imshow(Foreground(:,:,t)); title('Foreground');
    pause(0.10);
end
figure;imshow3D(Foreground);