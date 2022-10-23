clear
close all
addpath('.\datasets');
addpath('.\funs');

% try
%% ==================== Dataset ==========================
% dataname = 'MSRC'; omega=0.0009;
dataname = 'ORL_mtv'; omega=0.01;

fprintf('----------- Database:【 %s 】---------------\n',dataname);
load([dataname '.mat']);
try
    aa = 1;
    for i=1:length(X)-1
        aa = aa && size(X{i},2) == size(X{i+1},2);
    end
    if(aa)
        for i=1:length(X)
            X{i}=X{i}';
        end
    end
 catch
     X{1}=X1;X{2}=X2;X{3}=X3;
 end
try
    gt=double(gt);
catch
    gt=double(Y); gt=double(Y);
end
nV = size(X, 2);
[dv, nN] = size(X{1}'); %[样本维数，总样本数目]
nC = length(unique(gt)); 
Iter_max = 150;
pho_mu = 1.1;
%% ================ Parameter Setting =========================== 

resultFile = [ '.\results\' dataname '_bw' '.csv'];
All_lambda = [1/sqrt(nC*nN) 0.01 0.1 1  5 10 50 100];%
All_r = [5:10];%[3:10];
All_p = [1:-0.1:0.1];%0.1;%0.7;%

%% ================ Initializing ================================
tic
N=2;
opt1. style = 1;
opt1. IterMax = 150;
opt1. toy = 0;
opt1. k = 10;
anchor_rate = 0.5;
% 初始化
D = cell(1, nV); 
S = cell(1, nV);
Tri = cell(1, nV);
[~, C] = FastmultiCLR(X,nC,anchor_rate, opt1);
nM = floor(nN*anchor_rate);
for v = 1: nV
    Tri{v} = diag(sum(C{v}));
    S{v}=C{v}*pinv(Tri{v})*(C{v})';
end

% initialize D (distance)
for v = 1: nV
    D{v} = sqrt(1./(1+(S{v}./omega).^(4))); fun = ['bw'];
    D{v} = D{v}-diag(diag(D{v})); %对角置零
end
time1 = toc;

for r = All_r
for lambda = All_lambda
for p = All_p

tic
% initialize Y J Q H
Y_old = zeros(nN, nC);
Y = cell(1, nV);
J = cell(1, nV);
Q = cell(1, nV);
H = cell(1, nV);
for i = 1:nN
    Y{1}(i,mod(i, nC)+1) = 1;
end
% for i = 1:nN
for v = 1:nV
    Y{v} = Y{1};
    J{v} = zeros(nN, nC);
    Q{v} = zeros(nN, nC);
    H{v} = zeros(nN, nC);
end
alpha = ones(1,nV)./nV;
mu = 0.0001; max_mu = 12e12;
%% ================= Iterative Update ===========================
obj = [1];
flag = 1;
iter = 1;
while flag == 1  
    % Solving Y
    for v = 1:nV
        H{v} =  J{v} - Q{v}/mu;
        for i = 1:nN 
            M = (2*(alpha(v)^r)*(D{v}(i,:)*Y{v}) - mu*H{v}(i,:))';%*lambda
            [~, m] = min(M);
            Y{v}(i,:) = 0;
            Y{v}(i,m) = 1;
        end
    end
    % Solving J --Schatten p_norm
    for v =1:nV
        QQ1{v} = Y{v} + Q{v}./mu;
    end
    Q_tensor = cat(3,QQ1{:,:});
    Qg = Q_tensor(:);
    sX=[nN, nC, nV];
    [myj, ~] = wshrinkObj_weight_lp(Qg,ones(1, nV)'.*(1*lambda/mu),sX, 0,3,p);
    J_tensor = reshape(myj, sX);
    for v=1:nV
        J{v} = J_tensor(:,:,v);
        Q{v} = Q{v} + mu*(Y{v}-J{v});
    end
    
    mu = min(pho_mu*mu, max_mu);
    
    % Solving alpha 
    o = 0;
    h = zeros(1, nV);
    h_sum = 0;
    for v = 1:nV
        h(v) = (trace(Y{v}'*D{v}*Y{v}))^(1/(1-r));
        h_sum = h_sum + h(v);
    end
    alpha = h/h_sum;
    
    for v = 1:nV
        o = o + alpha(v)^r*trace(Y{v}'*D{v}*Y{v});
    end
    obj=[obj o + lambda*cal_tensorSp(Y,p)];
    
    obj2(iter)=abs(obj(iter+1)-obj(iter));
    % ShouLian
    if  obj2(iter)<1e-8 || iter == Iter_max
        flag = 0;
    end
    iter = iter + 1;
end
plot(obj)
time = toc;
time = time +time1;
%% ================== Perfermance Calculate=======================
Y_sum = zeros(nN, nC);

for v=1:nV
    Y_sum = Y_sum + alpha(v)^r*Y{v};                                  
end
[~, label] = max(Y_sum');
result = ClusteringMeasure(gt,label);

if ~exist(resultFile,'file')
   fid = fopen(resultFile,'w');
   fprintf(fid,'ACC,MIhat,Purity,P,R,F,RI,omega,lambda,r,p,time,iter\n');
   fclose(fid);
end
fid = fopen(resultFile,'a');
fprintf(fid,'%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.5f,%.4f,%.1f,%.1f,%.2f,%d,',result,omega,lambda,r,p,time,iter);
fprintf(fid,'%s \n',fun);
fprintf('lambda=%.4f,r=%.1f,p=%.1f,time(s)=%.2f,result=%.4f,%.4f,%.4f,iter=%d omega=%.5f\n',lambda,r,p,time,result(1:3),iter, omega);
fclose(fid);

end
end
end