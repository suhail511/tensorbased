clear all; clc;
addpath Yall1;
addpath Data;
addpath('./Tensor_Toolbox');

load Person.mat
b   =   0.95;
q   =   size(I,2);
% frame dimension
nrow = imSize(1);
ncol = imSize(2);

numTrain    = size(DataTrain,2);
mu0         = mean(DataTrain,2);
MTrain      = DataTrain-repmat(mu0,1,numTrain);
M           = I-repmat(mu0,1, size(I,2));
%% "Tensorifying" the Data
for i = 1:ncol
    tensorMTrain(1:nrow,i,:) = MTrain((i-1)*nrow+1:i*nrow,:);
end
for i = 1:ncol
	tensorM((1:nrow),i,:) = M((i-1)*nrow+1:i*nrow,:);
end

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
    
    % Projection matrices
    U{unf}              = U_cell0{unf}*U_cell0{unf}';
end
S0 = S(1:rhat{1},1:rhat{2},1:rhat{3});
lowRankTrain = nmode_product(S0,U_cell0{1},1); lowRankTrain = nmode_product(lowRankTrain,U_cell0{2},2); lowRankTrain = nmode_product(lowRankTrain,U_cell0{3},3);
%% Figure
close all;
figure;
set(gcf, 'units','normalized','outerposition',[0 0 1 1]); % Full screen.
for t=1:numTrain/5
    for i = 1 : ncol
        thisFrame(:,i) = uint8((DataTrain((i-1)*nrow+1:i*nrow,t))); 
        for j = 1 : nrow
            lowRank(j,i) = uint8(lowRankTrain(j,i,t) + mu0((i-1)*nrow +j));
        end
    end
    subplot(221); imshow(thisFrame); caption = sprintf('Training frame original %4d', t); title(caption);
    subplot(222); imshow(lowRank); caption = sprintf('Training frame Lowrank%4d', t); title(caption);
    pause(0.10);
end