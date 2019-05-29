clear all; clc;close all;clc;
addpath Yall1;
TrainPath = 'SABS\Train\NoForegroundDay\'; numTrain = 800;
VideoPath = 'SABS\Test\Bootstrap\'; videolength = 300;
nrow = 120; ncol = nrow * 160/120;% frame dimension


addpath(TrainPath);
addpath(VideoPath);

%% Reading the files

%Background
srcFiles = dir(strcat(TrainPath,'*.png'));
if length(srcFiles) > numTrain
    srcFiles = srcFiles(1:numTrain);
end
DataTrain = zeros([nrow*ncol length(srcFiles)]);
tensorDataTrain = zeros(nrow,ncol,numTrain);
tensorMTrain    = zeros(nrow,ncol,numTrain);
tensorM         = zeros(nrow,ncol,numTrain);
for i = 1 : length(srcFiles)
    filename = strcat(TrainPath,srcFiles(i).name);
    I1 = double(rgb2gray(imread(filename)));
    I1 = imresize(I1, [nrow NaN]);
    %imshow(I); title('Image ',i);
    DataTrain(:,i) = reshape(I1,[nrow*ncol 1]);
    tensorDataTrain(:,:,i) = I1;
end

b = 0.95;

mu0         = mean(DataTrain,2);
mu0_tensor  = mean(tensorDataTrain,3);
MTrain      = DataTrain-repmat(mu0,1,numTrain);
for t=1:numTrain
    tensorMTrain(:,:,t)= tensorDataTrain(:,:,t) - mu0_tensor;
end
%% SVD
[Usvd, Sig, Vsvd] = svd(1/sqrt(numTrain)*MTrain,0);

%Keeping b% energy
evals1      = diag(Sig).^2;
energy1     = sum(evals1);
cum_evals1  = cumsum(evals1);
ind01       = find(cum_evals1 < b*energy1);
rhat1       = min(length(ind01),round(numTrain/10));
lambda_min1 = evals1(rhat1);
U1          = Usvd(:, 1:rhat1);
V1          = Vsvd(1:rhat1 , :);
Sig1        = Sig(1:rhat1,1:rhat1);

Mtrain_lowrank_reprocs = U1 * Sig1 * V1;

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
    T{unf}              = U_cell0{unf} * (U_cell0{unf})';
end

U2 = kron(T{1},T{2});
U = U2*U1;

S0 = S(1:rhat{1},1:rhat{2},1:rhat{3});
MTrain_lowrank_tensor = nmode_product(S0,U_cell0{1},1); 
MTrain_lowrank_tensor = nmode_product(MTrain_lowrank_tensor,U_cell0{2},2); 
MTrain_lowrank_tensor = nmode_product(MTrain_lowrank_tensor,U_cell0{3},3);

%% Displaying Backgrounds after low-rank approximation

close all;
figure;
set(gcf, 'units','normalized','outerposition',[0 0 1 1]); % Full screen.
for t=1:videolength
    Mtrain_lowrank_reprocs(:,t) = Mtrain_lowrank_reprocs(:,t) + mu0;
    for i = 1 : ncol
        reprocs(:,i) = uint8(Mtrain_lowrank_reprocs((i-1)*nrow+1:i*nrow,t));
    end
    original = uint8(tensorDataTrain(:,:,t));
    tensor   = uint8(MTrain_lowrank_tensor(:,:,t) + mu0_tensor );
    subplot(221); imshow(original); caption = sprintf('Original Training Background');
    subplot(222); imshow(reprocs); caption = sprintf('ReProCS Low rank Training Frame %4d', t); title(caption);
    subplot(223); imshow(tensor); caption = sprintf('Tensor Low rank Training Frame %4d', t); title(caption);
    pause(0.04);
end