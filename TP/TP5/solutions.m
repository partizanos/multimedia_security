clc
clear

%% Parameters
densities = [0.1, 0.3];
%sigma_noises = [sqrt(50), sqrt(100)];
sigma_noises = [50, 100];
gammas = [1, 5];
x0 = imread('cameraman.tif');
J = 100;

NB = zeros(4,8);
B = zeros(4,8);
ir = 1;
for sigma_noise = sigma_noises
    for density = densities
        N = int32(numel(x0)*density);
        for gamma = gammas
            
            x = repmat(x0,1,1,J);
            z = randn(size(x))*sigma_noise;
            v = double(x) + z;
            w = zeros(size(x));
            v = repmat(v,1,1,2);
            for i=1:J
                w0 = 2*gamma*(rand(size(x0))<0.5)-gamma;
                w0(randperm(numel(x0),numel(x0)-N)) = 0;
                w(:,:,i) = w0;
            end
            v(:,:,1:J) = v(:,:,1:J) + w;
            
            wrNB = v - double(repmat(x0,1,1,2*J));
            rhosNB = linear_corr(repmat(w,1,1,2), wrNB, N);
            
            taus = min(rhosNB(:)):0.1:max(rhosNB(:));
            pf = zeros(1,size(taus,2));
            pm = zeros(1,size(taus,2));
            pd = zeros(1,size(taus,2));
            ip = 1;
            for tau=taus
                % 0 if no WM is detected and 1 if WM is detected
                detectNB = rhosNB>tau;
                pf(ip) = sum(detectNB(J+1:2*J))/J;
                pm(ip) = sum((1-detectNB(1:J)))/J;
                pd(ip) = 1-pm(ip);
                ip = ip+1;
            end
            figure('Name',sprintf('Non-Blind ROC_tau s%s d%s g%s',num2str(sigma_noise), num2str(density), num2str(gamma)));
            plot(taus,pf,taus,pm,taus,pd);
            legend('pf(Tp)','pm(Tp)', 'pd(Tp)');
            
            figure('Name',sprintf('Non-Blind ROC_pfpm s%s d%s g%s',num2str(sigma_noise), num2str(density), num2str(gamma)));
            plot(pm,pf);
            legend('pf(pm)');
            
            NB(3,ir) = mean(rhosNB(1:J));
            NB(4,ir) = var(rhosNB(1:J));
            NB(1,ir) = mean(rhosNB(J+1:2*J));
            NB(2,ir) = var(rhosNB(J+1:2*J));
            
            figure();
            histogram(rhosNB(1:J));
            hold on;
            histogram(rhosNB(J+1:2*J));
            legend(sprintf('NB s%s d%s g%s',num2str(sigma_noise), num2str(density), num2str(gamma)));
            
%             for t=-1:0.1:gamma+1
%             end
            
            xr = zeros(size(v));
            for i=1:size(v,3)
                xr(:,:,i) = conv2(v(:,:,i), ones(3)/9, 'same');
            end
            wrB = v - xr;
            rhosB = linear_corr(repmat(w,1,1,2), wrB, N);
            
            taus = min(rhosB(:)):0.1:max(rhosB(:));
            pf = zeros(1,size(taus,2));
            pm = zeros(1,size(taus,2));
            pd = zeros(1,size(taus,2));
            ip = 1;
            for tau=taus
                % 0 if no WM is detected and 1 if WM is detected
                detectB = rhosB>tau;
                pf(ip) = sum(detectB(J+1:2*J))/J;
                pm(ip) = sum((1-detectB(1:J)))/J;
                pd(ip) = 1-pm(ip);
                ip = ip+1;
            end
            figure('Name',sprintf('Blind ROC_tau s%s d%s g%s',num2str(sigma_noise), num2str(density), num2str(gamma)));
            plot(taus,pf,taus,pm,taus,pd);
            legend('pf(Tp)','pm(Tp)', 'pd(Tp)');
            
            figure('Name',sprintf('Blind ROC_pfpm s%s d%s g%s',num2str(sigma_noise), num2str(density), num2str(gamma)));
            plot(pm,pf);
            legend('pf(pm)');
            
            B(3,ir) = mean(rhosB(1:J));
            B(4,ir) = var(rhosB(1:J));
            B(1,ir) = mean(rhosB(J+1:2*J));
            B(2,ir) = var(rhosB(J+1:2*J));
            
            ir = ir+1;
            
        end
    end
end

NB
B

%% Non-blind detection

function [rho] = linear_corr(w1,w2,N)
rho = zeros(1,size(w1,3));
for i=1:size(w1,3)
    w1i = w1(:,:,i);
    w2i = w2(:,:,i);
    rho(i) = dot(w1i(:),w2i(:))/double(N);
end
end