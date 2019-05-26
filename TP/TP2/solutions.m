%% 
%ex1
LB = imread('liftingbody.png');
LBf = LB(:);
N = size(LBf,1);
P = randperm(N);
LBp = reshape(LB(P),size(LB));

figure('Name','liftingbody&permuted');
subplot(1,2,1), imshow(LB);
subplot(1,2,2), imshow(LBp);

figure('Name','liftingbody&permuted_histograms');
subplot(1,2,1), histogram(LB);
subplot(1,2,2), histogram(LBp);
%% 

%% 
%ex3
LBp_bl = block_loss(LBp);
LB_bl(P) = LBp_bl;
LB_bl = reshape(LB_bl, size(LB));

figure('Name','liftingbody&permutedBL');
subplot(1,3,1), imshow(LB);
subplot(1,3,2), imshow(LBp_bl);
subplot(1,3,3), imshow(LB_bl);
%% 

%% 
%ex4
Vn = [1,5,10,15];
psnrV = [];
figure('Name','liftingbody_noisy');
ind = 1;
for noise=Vn
    noiseV = randi([0,1], size(LB))*2*noise-noise;
    LB_noisy = double(LB)+double(noiseV);
    LB_noisy(LB_noisy>255) = 255;
    LB_noisy(LB_noisy<0) = 0;
    LB_noisy = uint8(LB_noisy);
    psnrV(ind) = psnr(LB_noisy,LB);
    subplot(1,size(Vn,2),ind), imshow(LB_noisy);
    title(strcat('psnr=',num2str(psnrV(ind))));
    ind = ind+1;
end
figure('Name','PSNR vs. Noise');
plot(Vn,psnrV);
legend('psnr vs. noise');
%% 

%% 
%ex2 - Data hidding
[n,m,e] = computer;
if e == 'L'
    endianness = 'little';
else
    endianess = 'big';
end
fprintf('System is working on %s endian', endianness);

L = imread('lena.png');
B = imread('baboon.png');
LB = insert_secret(L,B);
BR = recover_secret(LB);
figure('Name','BaboonHiddenInLena')
subplot(1,2,1), imshow(L);
title('Lena')
subplot(1,2,2), imshow(LB);
title('LenaWithBaboon')

figure('Name','SecretBaboon')
subplot(1,2,1), imshow(B);
title('Baboon')
subplot(1,2,2), imshow(BR);
title('BaboonRecovered')
%% 

%% 
%ex2
function [image_bl]=block_loss(image)
[X, Y] = size(image);
x01 = randi([1,X], 1, 2);
y01 = randi([1,Y], 1, 2);
if x01(1)>=x01(2)
    x01(2) = randi([x01(1)+1,X], 1, 1);
end
if y01(1)>=y01(2)
    y01(2) = randi([y01(1)+1,X], 1, 1);
end
image_bl = image;
image_bl(x01(1):x01(2),y01(1):y01(2)) = 0;
end
%% 

%% 
%ex1 - data hidding
function [CS]=insert_secret(C,S)
Cr_mask = bin2dec('11111000');
Cg_mask = bin2dec('11111100');
Cb_mask = bin2dec('11100000');
C_mask = uint8(cat(3,repmat(Cr_mask,size(C,1), size(C,2)),repmat(Cg_mask,size(C,1), size(C,2)),repmat(Cb_mask,size(C,1), size(C,2))));
disp(size(C_mask));
disp(size(C));
Sr_mask = bitxor(Cr_mask, bin2dec('11111111'));
Sg_mask = bitxor(Cg_mask, bin2dec('11111111'));
Sb_mask = bitxor(Cb_mask, bin2dec('11111111'));
S_mask = uint8(cat(3,repmat(Sr_mask,size(C,1), size(C,2)),repmat(Sg_mask,size(C,1), size(C,2)),repmat(Sb_mask,size(C,1), size(C,2))));
if sum(size(C)==size(S))~= 3
    error('images sizes do not match')
end
% Shits and move the bit positions of S accordingly to the table
S(:,:,1) = bitshift(S(:,:,1),-5);
S(:,:,3) = bitshift(S(:,:,2),-3);
S(:,:,2) = bitshift(S(:,:,3),-4);
S3_shiftmask = bin2dec('11111100');
S2_shiftmask = bin2dec('00000011');
S(:,:,3) = bitor(bitand(S(:,:,2),S2_shiftmask),bitand(S(:,:,3),S3_shiftmask));
S(:,:,2) = bitshift(S(:,:,2),-2);
% Puts the information from S into C at the right positions
CS = bitor(bitand(C_mask,C),bitand(S_mask,S));
end

function [S]=recover_secret(CS)
S = CS;
S(:,:,1) = bitshift(S(:,:,1),5);
S(:,:,2) = bitshift(S(:,:,2),2);
S(:,:,2) = bitor(S(:,:,2),bitand(S(:,:,3),bin2dec('00000011')));
S(:,:,2) = bitshift(S(:,:,2),4);
S(:,:,3) = bitshift(bitand(S(:,:,3),bin2dec('00011100')),3);
temp = S(:,:,3);
S(:,:,3) = S(:,:,2);
S(:,:,2) = temp;
end