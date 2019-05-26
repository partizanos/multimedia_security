clc
clear
%% 
%Part 1
fprintf('1 - Elements of Detection Theory\nstart: %s\n\n', datetime('now'));

fprintf('P(-1<Y<=1) = %s\n',num2str(normal_proba(-1,1,0,3)));
fprintf('P(Y>3.5) = %s\n',num2str(normal_proba(3.5,Inf,0,3)));
fprintf('P(-2<X<=2) = %s\n',num2str(normal_proba(-2,2,0,1)));
fprintf('P(X>2.5) = %s\n',num2str(normal_proba(2.5,Inf,0,1)));

fprintf('-----\n');

fprintf('P(X>40) = %s\n',num2str(normal_proba(40,Inf,30,11)));
fprintf('P(X<=15) = %s\n',num2str(normal_proba(-Inf,15,30,11)));
fprintf('P(20<X<=40) = %s\n',num2str(normal_proba(20,40,30,11)));

fprintf('-----\n');

fprintf('P(|X|<10)=P(-10<X<10) = P(-10/s<X/s<10/s) = 0.2\nSo phi(10/s) = 0.5+0.2/2 = 0.6\n');
fprintf('Then 10/s = inv_phi(0.6) = %s\nAnd s = 10/inv_phi(0.6) = %s\n\n',num2str(norminv(0.6)),num2str(10/norminv(0.6)));

fprintf("By variable substitution x'=sqrt(2)x in integral of erfc(n/sqrt(2)), one can get:\nerfc(n/sqrt(2)) = 2Q(n)\nThen Q(n) = 1/2*erfc(n/sqrt(2))\n\n");
%% 

%% 
% Part 2
fprintf('2 - Bayesian 2-class classification\nstart: %s\n\n', datetime('now'));

fprintf('p = p(H0) and 1-p = p(H1) are the 2 general prior probabilities.\n');
fprintf('pm = p(H0|H1) = pH1(X<tau)\n');
fprintf('Separation threshold tau is such that pH0(x)/pH1(x) >< (1-p)/p\n');
fprintf('exp(-x^2/(2s^2))/exp(-(x-mu1)^2/(2s^2)) >< (1-p)/p\n');
fprintf('exp((-x^2+(x-mu1)^2)/(2s^2)) >< (1-p)/p\n');
fprintf('-x^2+(x-mu1)^2 >< 2s^2*ln((1-p)/p)\n');
fprintf('mu1^2-2*mu1*x >< 2s^2*ln((1-p)/p)\n');
fprintf('tau = mu1/2-s^2/mu1*ln((1-p)/p)\n\n');
fprintf('mu1 = 1 and s = 1 so:\ntau = 1/2-ln((1-p)/p)\nSo pm = phi(tau - mu1) = phi(-ln((1-p)/p)-1/2)\nAnd pd = Q(-ln((1-p)/p)-1/2)\n');

fprintf('-----\n');

fprintf('From previous:\npm = phi(tau-mu1) = phi(-s^2/mu1*ln((1-p)/p)-mu1/2)\npd = Q(tau-mu1) = Q(-s^2/mu1*ln((1-p)/p)-mu1/2)\npf = Q(tau) = Q(mu1/2-s^2/mu1*ln((1-p)/p))\n\n');
fprintf('In matlab, phi is normcdf and Q is qfunc so we can plot ROCs:\npm = normcdf(tau-mu1) = normcdf(qfuncinv(pf)-mu1)\n');
figure('Name','ROC: pm, pd, pf');
X = -2:0.01:6;
Y0 = normcdf(X);
Y1 = normcdf(X-1);
Y2 = normcdf(X-2);
Y3 = qfunc(X);
plot(X,Y0,X,Y1,X,Y2,X,Y3);
legend('pm : mu1=0','pm : mu1=1','pm : mu1=2','pf');

figure('Name','ROC: pm vs pf');
X = 0:0.01:1;
Y0 = normcdf(qfuncinv(X));
Y1 = normcdf(qfuncinv(X)-1);
Y2 = normcdf(qfuncinv(X)-2);
plot(X,Y0,X,Y1,X,Y2);
legend('mu1=0','mu1=1','mu1=2');

fprintf('As expected, for larger mu1, the separation and detection is better. If mu1=0, there is no detection between hypothesis 0 of 1.\n');

fprintf('-----\n');

fprintf('In this situation:\nH0: Y = W and p(H0) = p(X=0) = 1-p\nH1: Y = V+W and p(H1) = p(X=1) = p\n\n');
fprintf('Separation threshold for MAP is such that pH0(x)/pH1(x) >< p/(1-p)\n');
fprintf('exp(-x)/(x*exp(-x)) >< p/(1-p)\n');
fprintf('1/x >< p/(1-p)\n');
fprintf('So tau = (1-p)/p such that if x<tau one chooses H0, else one chooses H1.\n\n');
fprintf('perr = pH0(x>tau)*p(H0)+pH1(x<tau)*p(H1)\n');
fprintf('So perr = exp(-tau)*(1-p)+(1-tau*exp(-tau)-exp(-tau))*p by integration by parts,\n');
fprintf('Then perr = exp(-tau)-p*(tau+2)*exp(-tau)+p.\nperr = p*(1-exp(-tau)).\n');
%% 

fprintf('\nend: %s\n',datetime('now'));

%% 
%functions
function [a,b] = rescale_01(u,v,mu,sigma)
a = (u-mu)/sigma;
b = (v-mu)/sigma;
end

function p = normal_proba(u,v,mu,sigma)
[a,b] = rescale_01(u,v,mu,sigma);
p = normcdf(b)-normcdf(a);
end