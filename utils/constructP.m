
%% 分布对齐 学习一个矩阵P domain invariant yet discriminative subspace
%% ZΩZ'P = αZHZ'Pθ 
%% Ω = Q^(all) + δI
%% Qall = Qcc + lambda * M - beta * Q_c1 - gamma * Q_d + eta * L;
%% Qcc 分类器误差项；M：MMD matrix；Q_c1: other K-1 classes; Q_d: cross-domain error; L：Discriminative matrix；参数：γ,λ,η,δ

function P = constructP(Xs,Ys,Xt,Yt_pseudo, W,options)

if nargin < 6
    error('Algorithm parameters should be set!');
end
if ~isfield(options,'beta')
    options.beta = 1;
end
if ~isfield(options,'lambda')
    options.lambda = 1;
end
if ~isfield(options,'gamma')
    options.gamma = 0.5;
end
if ~isfield(options,'eta')
    options.eta = 0.0001;
end
if ~isfield(options,'sigma')
    options.sigma = 0.1;
end

beta = options.beta;
lambda = options.lambda;
gamma = options.gamma;
eta = options.eta;
sigma = options.sigma;
reduced_dim = options.ReducedDim;
%fprintf('lambda=%0.3f, gamma=%0.3f, sigma=%0.3f, eta=%0.3f\n', lambda,gamma, sigma, eta);
ns = size(Xs,1);
nt = size(Xt,1);
n = ns + nt;
C = length(unique(Ys));
d = size(Xs, 2);

% Construct MMD matrix
% margin distribution MMD matrix
e = [1/ns*ones(ns,1);-1/nt*ones(nt,1)];
M = lambda * (e*e')* C;  % marginal MMD
Qnew = zeros(ns+nt);

if ~isempty(Yt_pseudo)
    % 筛选出label不为-1的样本
    idx = find(Yt_pseudo ~= -1);
    Xt = Xt(idx(1,:), :);
    Yt_pseudo = Yt_pseudo(idx);
    nt = size(Xt, 1);
    n = ns + nt;
    idx_s = 1:ns;
    W = W([idx_s, ns+idx], [idx_s, ns+idx]);
    e = [1/ns*ones(ns,1); -1/nt*ones(nt,1)];
    M = lambda * (e*e')* C;
    
    % bulid Qc
    for c = reshape(unique(Ys),1,C)
        e = zeros(n,1);
        e(Ys==c) = 1/length(find(Ys==c));
        e(ns+find(Yt_pseudo==c)) = -1/length(find(Yt_pseudo==c));
        e(isinf(e)) = 0;
        nc = length(find(Ys==c)) + length(find(Yt_pseudo == c));
        M = M + lambda * nc * (e*e');
    end
    
    tp = zeros(nt, C);
    for c = 1: C
        tp(Yt_pseudo==c, c) = 1;
    end
    tp2 = tp*diag(1./(1e-4+sum(tp)));
    tp3 = tp2*tp';
    Qcc_t = eye(nt) - tp3;
    
    %Construct Qnew
    Qnew = zeros(ns+nt);
    for c = 1:C      
        nsc = length(find(Ys==c));
        nsk = ns - nsc;
        ntc = length(find(Yt_pseudo == c));
        ntk = nt - ntc;
        %Construct Qck
        Qck = zeros(ns+nt);
        e = zeros(ns,1);
        e(Ys==c) = 1/nsc;
        e(Ys~=c) = -1/nsk;
        e(isinf(e)) = 0;
        Qsck = nsc*(e*e');
        f = zeros(nt,1);
        f(Yt_pseudo==c) = 1/ntc;
        f(Yt_pseudo~=c) = -1/ntk;
        f(isinf(f)) = 0;
        Qtck = ntc*(f*f');
        Qck(1:ns,1:ns) = Qsck;
        Qck(ns+1:end,ns+1:end) = Qtck;
        %Construct Qstck
        g = zeros(ns+nt,1);
        g(Ys==c) = 1/nsc;
        g(ns+find(Yt_pseudo~=c)) = -1/ntk;
        g(isinf(g)) = 0;
        Qstck = nsc*(g*g');
        %Construct Qtsck
        h = zeros(ns+nt,1);
        h(Ys~=c) = 1/nsk;
        h(ns+find(Yt_pseudo==c)) = -1/ntc;
        h(isinf(h)) = 0;
        Qtsck = ntc*(h*h');
        Qnew = Qnew + beta*Qck + gamma*Qstck + gamma*Qtsck;
    end
end

%% Qcc matrix contruction
Qcc = zeros(ns + nt);
tp = full(sparse(1:ns,  double(Ys), 1));  %% (ns, c)
tp2 = tp*diag(1./sum(tp)); 
tp3 = tp2*tp';
Qcc(1:ns,1:ns) = eye(ns) - tp3;
if ~isempty(Yt_pseudo)
    Qcc(ns+1:end,ns+1:end) = Qcc_t;
end

% bulid LPP matrix
W = double(W);
D = diag(sum(W,2));
L = D - W;

%Construct new Qall
Qall = Qcc + M - Qnew + eta * L;

% Construct centering matrix
H = eye(n)-1/(n)*ones(n,n);

X = [Xs;Xt];
Omega = X' * Qall * X + sigma * eye(d);
Omega = (Omega+Omega')/2;
S1 = X'*H*X + 0.0001 * eye(d);

opts.disp = 0;
opts.tol = 1e-4;
[P,~] = eigs(double(Omega), double(S1), reduced_dim,'sa',opts); 
P = real(P);
 for i = 1:size(P,2) 
     if (P(1,i)<0) 
         P(:,i) = P(:,i)*-1;
     end
 end
