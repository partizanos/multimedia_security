warning off all;
close all
clear
clc

%% --------- 0-Constants ----------------------------------------------- %%
save_dir = '.\solutions';
sources_dir = '.\images\dct_db';
[status,msg,msgID] = mkdir(save_dir);  % Creating directory for output solutions
if status == 0
    fprintf(strcat(msg, ' . Error ID: ' ,msgID));
end

%% --------- I-Introduction -------------------------------------------- %%

fprintf('%s: Part I Introduction - start \n', datestr(now));

P = imread('peacock.jpg');
fprintf('Size of peacock.jpg:\n');
disp(size(P));
figure('Name','peacock_histogram');
histogram(P);
saveas(gcf,strcat(save_dir,'\peacock_hist.png'));
PG = rgb2gray(P);
DPG = im2double(PG);
GM = mean(DPG(:));
GV = var(DPG(:));
fprintf('Global mean : %s \n Global variance : %s \n', num2str(GM), num2str(GV));
bmean = @(block_struct) mean(block_struct.data(:));
LM = blockproc(DPG,[3 3],bmean);
bvar = @(block_struct) var(block_struct.data(:));
LV = blockproc(DPG,[3 3],bvar);

figure('Name','peacock_lmeans');
imshow(LM);
saveas(gcf,strcat(save_dir,'\peacock_lmeans.png'));

figure('Name','peacock_lvar');
imshow(LV);
saveas(gcf,strcat(save_dir,'\peacock_lvars.png'));

fprintf('%s: Part I Introduction - end \n', datestr(now));

%% --------- II-Noise -------------------------------------------------- %%

fprintf('%s: Part II Noise - start \n', datestr(now));

Co = imread('peacock.jpg');
Cd = im2double(imread('peacock.jpg'));
error = mse(Co,Cd);
fprintf('error between original image and double mapped image: %s \n', num2str(error));

% ------------ Metrics ------------
% PSNR = 20*log10(alpha/sigmaz) = 20*log10(alpha) - 20*log10(sigmaz)
% so sigmaz = 10^((20*log10(alpha) - PSNR)/20)

psnr = [10,20,30,40];
sigmaz = sqrt(255^2./10.^(psnr./10));
Co = double(rgb2gray(Co));
[N,M] = size(Co);
Cn10 = uint8(Co+awgn(N,M,sigmaz(1)));
Cn20 = uint8(Co+awgn(N,M,sigmaz(2)));
Cn30 = uint8(Co+awgn(N,M,sigmaz(3)));
Cn40 = uint8(Co+awgn(N,M,sigmaz(4)));
figure('Name','peacock_awgn_10dB');
imshow(Cn10);
saveas(gcf,strcat(save_dir,'\peacock_awgn_10dB.png'));
figure('Name','peacock_awgn_20dB');
imshow(Cn20);
saveas(gcf,strcat(save_dir,'\peacock_awgn_20dB.png'));
figure('Name','peacock_awgn_30dB');
imshow(Cn30);
saveas(gcf,strcat(save_dir,'\peacock_awgn_30dB.png'));
figure('Name','peacock_awgn_40dB');
imshow(Cn40);
saveas(gcf,strcat(save_dir,'\peacock_awgn_40dB.png'));
figure('Name','peacock_awgn_10dB_histogram');
histogram(Cn10);
saveas(gcf,strcat(save_dir,'\peacock_awgn_10dB_histogram.png'));
figure('Name','peacock_awgn_20dB_histogram');
histogram(Cn20);
saveas(gcf,strcat(save_dir,'\peacock_awgn_20dB_histogram.png'));
figure('Name','peacock_awgn_30dB_histogram');
histogram(Cn30);
saveas(gcf,strcat(save_dir,'\peacock_awgn_30dB_histogram.png'));
figure('Name','peacock_awgn_40dB_histogram');
histogram(Cn40);
saveas(gcf,strcat(save_dir,'\peacock_awgn_40dB_histogram.png'));

Cnsp40 = Co;
ratio = 41;
% finding sp for 40dB (by adding more and more disturbed pixels until the desired psnr)
s = 0;
p = 0;
while ratio>40
    for i=1:N
        sp = saltpepper(0.1*min(1,(ratio-40)),0.1*min(1,(ratio-40)));
        if not(isnan(sp))
            Cnsp40(i)=sp;
        end
    end
    s = s+0.1*min(1,(ratio-40));
    p = p+0.1*min(1,(ratio-40));
    ratio = fpsnr(Co,Cnsp40);
    disp(ratio);
end
figure('Name','peacock_sp_40dB');
imshow(uint8(Cnsp40));
saveas(gcf,strcat(save_dir,'\peacock_sp_40dB.png'));
fprintf('%s db obtained for s=%s & p=%s \n',num2str(ratio), num2str(s), num2str(p));

fprintf('%s: Part II Noise - end \n', datestr(now));

%% --------- III Identification ----------------------------------------- %%

fprintf('%s: Part III Identification - start \n', datestr(now));

% loading gray images
[images] = togray(sources_dir);
nimages = size(images, 1);
[N,M] = size(images(1,:,:));

% global means
gm = NaN(nimages,1);
for i = 1:nimages
    image = images(i,:,:);
    image = double(image);
    gm(i) = mean(image(:));
end

% binary descriptor vectors
ldesc = N*M/32/32;
descriptors = NaN(nimages,ldesc);
for i = 1:nimages
    image = images(i,:,:);
    image = squeeze(double(image));
    LM = blockproc(image,[32 32],bmean);
    Hashes = LM>gm(i);
    descriptors(i,:) = Hashes(:);
end

% Inter class distance
desc_conf = descriptors';
desc_conf(desc_conf==0) = -1;
Z = desc_conf'*desc_conf;
Z = (Z-ldesc*ones(size(Z)))/(-2*ldesc);
figure('Name','histogram_inter');
histogram(Z/sum(Z(:)));
saveas(gcf,strcat(save_dir,'\histogram_inter.png'));

% copy and distort all images with PSNR of 35dB
[dimages] = distort(images,35,save_dir);

% global means
dgm = NaN(nimages,1);
for i = 1:nimages
    image = dimages(i,:,:);
    image = double(image);
    dgm(i) = mean(image(:));
end

% binary descriptor vectors
ddescriptors = NaN(nimages,ldesc);
for i = 1:nimages
    image = squeeze(dimages(i,:,:));
    image = double(image);
    DLM = blockproc(image,[32 32],bmean);
    Hashes = DLM>dgm(i);
    ddescriptors(i,:) = Hashes(:);
end

% Intra class distance
Pb_intra = NaN(nimages,1);
for i=1:nimages
    [~,proba] = hamming_pb(descriptors(i,:),ddescriptors(i,:));
    Pb_intra(i) = proba;
end
figure('Name','histogram_intra');
histogram(Pb_intra/sum(Pb_intra(:)));
saveas(gcf,strcat(save_dir,'\histogram_intra.png'));

figure('Name','histograms');
h1 = histogram(Z);
hold on
h2 = histogram(Pb_intra);
h1.Normalization = 'probability';
h1.BinWidth = 0.01;
h2.Normalization = 'probability';
h2.BinWidth = 0.01;
saveas(gcf,strcat(save_dir,'\histograms.png'));

fprintf('%s: Part III Identification - end \n', datestr(now));

%% --------- Functions ------------------------------------------------- %%

% function that generates awgn
function [Noise] = awgn(N,M,sigma)
Noise = randn(N,M)*sigma;
end

%function that generates salt&pepper noise
function [sp] = saltpepper(p,q)
a=rand();
if a<=p
    sp = 0;
elseif a<=p+q
    sp = 255;
else
    sp = NaN;
end
end

% function that calculates the mse between 2 images x and y
function [error] = mse(x,y)
[N,M] = size(x);
if size(x) == size(y)
    e = (double(x)-double(y)).^2;
    error = sum(e(:))/N/M;
else
    error('Sizes incompatible.');
end
end

% function that determines the psnr between 2 images x and y
function [ratio] = fpsnr(x,y)
error = mse(x,y);
if isempty(find(class(x)~=class(y),1))
    if isfloat(x)
        alpha = 255;
    elseif isinteger(x)
        alpha = intmax(class(x));
    end
else
    error('Not the same data types.');
end
ratio = 10*log10(alpha^2/error);
end

% function that determines the Hamming distance h and probability of error
% Pb between 2 binary vectors u and v
function [h,proba] = hamming_pb(u,v)
diff = xor(u,v);
h = sum(diff(:));
proba = h/size(diff(:),1);
end

%function for loading gray images from directory d
function [images] = togray(d)
imagefiles = dir(d);
nfiles = length(imagefiles);
di = 1;
while ismember(imagefiles(di).name,{'.','..'})
    di = di+1;
end
addpath(imagefiles(di).folder)
[N,M,~] = size(imread(imagefiles(di).name));
images = NaN(nfiles,N,M);
nimages = nfiles;
for i=1:nfiles
   currentfilename = imagefiles(i).name;
   [~, ~, extension] = fileparts(imagefiles(i).name);
   if extension == '.'
       nimages = nimages-1;
   else
       currentimage = imread(currentfilename);
       if size(size(currentimage),2) == 3
           images(i,:,:) = rgb2gray(currentimage);
       else
           images(i,:,:) = currentimage;
       end
   end
end
images = images(di:di+nimages-1,:,:);
end

%function that takes all images of images one by one, makes a copy and distorts this
%copy with AWGN resulting in a PSNR of psnr
function [dimages] = distort(images,psnr,save_dir)
nimages = size(images,1);
dimages=images;
[status,msg,msgID] = mkdir(strcat(save_dir,'\distorted\'));  % Creating directory for output solutions
if status == 0
    fprintf(strcat(msg, ' . Error ID: ' ,msgID));
end

for i=1:nimages
    currentimage = squeeze(images(i,:,:));
    sigma = 10.^((20*log10(255) - psnr)/20);
    [N,M] = size(currentimage);
    distortedimage = currentimage+awgn(N,M,sigma);
    dimages(i,:,:) = distortedimage;
    imwrite(uint8(distortedimage),strcat(save_dir,'\distorted\',num2str(i),'.png'));
end
end