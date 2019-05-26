clc
clear

%% Parameters
density = 0.5;
sigma_noise = 1;

%% Model
x = imread('liftingbody.png');
w = 2*(rand(size(x))<0.5)-1;
N = int32(numel(x)*density);
w(randperm(numel(x),numel(x)-N)) = 0;
y = double(x) + w;
z = randn(size(x))*sigma_noise;
v = y + z;

figure('Name','Original, WM, WM+attack');
subplot(1,3,1);
imshow(x);
title('Original');
subplot(1,3,2);
imshow(y,[0 255]);
title('Watermarked');
subplot(1,3,3);
imshow(v, [0 255]);
title('Watermarked & attacked')

%% Non-blind detection
wrNB = v - double(x);
rhoNB = linear_corr(w,wrNB,N);

%% Blind detection
xr = conv2(v, ones(3)/9, 'same');
wrB = v - xr;
rhoB = linear_corr(w,wrB,N);

%% Results and functions
fprintf('Linear correlations between the original and estimated WM:\nNon-Blind:%s\nBlind:%s',num2str(rhoNB),num2str(rhoB));

function [rho] = linear_corr(w1,w2,N)
rho = dot(w1(:),w2(:))/double(N);
end
