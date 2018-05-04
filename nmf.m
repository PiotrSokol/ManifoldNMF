function [W,D,H, loss] = nmf(V,rank, type)
    k = rank;
    maxiter = 500;
    rho = 1;
    loss = NaN(500,1);

    if nargin <3 
        type = 'admm_euclidean';
    end
    assert( strcmp(type,'admm_euclidean')| strcmp(type,'admm_manifold')...
    |strcmp(type,'pgrad_euclidean')|strcmp(type,'pgrad_manifold'))
switch type
    case 'admm_euclidean'
            fixed = [];
            free = setdiff(1:k, fixed);
            [W,d,H] = initialize(V,k);
            W = W*d;
%             d =1;
            H = H';
            X = W*d*H;
            
            Wplus = W;
            Hplus = H;
            alphaX = zeros(size(X));
            alphaW = zeros(size(W));
            alphaH = zeros(size(H));
        for i = 1:maxiter


            %H update
            H = (W'*W + eye(k)) \ (W'*X + Hplus + 1/rho*(W'*alphaX - alphaH));
            %W update
            P = H*H' + eye(k);
            Q = H*X'+ Wplus'+ 1/rho*(H*alphaX'- alphaW');
%             W = (P\(Q-P*W'))';
            W(:,free) = ( P(:,free) \ (Q - P(:,fixed)*W(:,fixed)') )';
            %X update
            X = (1+rho).^-1 .* (-alphaX + V + rho*W*H);
            % H_+ and W_+ update
            Hplus = max(H + 1/rho *alphaH,0);
            Wplus = max(W + 1/rho *alphaW,0);

            % dual update
            alphaX = alphaX + rho*(X - W*H);
            alphaH = alphaH + rho*(H - Hplus);
            alphaW = alphaW + rho*(W - Wplus);
            
            loss(i) = cost(V,W,1,H'); 
        end
        W = Wplus;
        H = Hplus;
    case 'admm_manifold'
        [W,D,H] = initialize(V,k);
        Wplus = max(W,0);
        Dplus = max(D,0);
        Hplus = max(H,0);
        X = W*D*H';
        LambdaW = zeros(size(W));
        LambdaD = zeros(size(D));
        LambdaH = zeros(size(H));
        LambdaX = zeros(size(X));
        
        optimizer = optimizer_factory(V,k, rho);
        for i = 1:10
            %% W update         
            optimizer.W.problem.cost = @(w)optimizer.W.cost(X,w,D,H,Wplus,LambdaW);
            optimizer.W.problem.egrad = @(w)optimizer.W.egrad(X,w,D,H,Wplus,LambdaW);
            W = rlbfgs(optimizer.W.problem, W, optimizer.opts);
            %% D update
            optimizer.D.problem.cost = @(d)optimizer.D.cost(X,W,d,H,Dplus,LambdaD);
            optimizer.D.problem.egrad = @(d)optimizer.D.egrad(X,W,d,H,Dplus,LambdaD);
            D = rlbfgs(optimizer.D.problem, D, optimizer.opts);
            %% H update
            optimizer.H.problem.cost = @(h)optimizer.H.cost(X,W,D,h,Hplus,LambdaH);
            optimizer.H.problem.egrad = @(h)optimizer.H.egrad(X,W,D,h,Hplus,LambdaH);
            H = rlbfgs(optimizer.H.problem, H, optimizer.opts);
            %% X update
            X = (1+rho)^(-1).*(V + rho*(W*D*H' - LambdaX));
            %% W_+, D_+, H_+
            Wplus = max(W+LambdaW, 0); 
            Dplus = max(D + LambdaD, 0);
            Hplus = max(H + LambdaH, 0);
            %% Dual update
            LambdaW = LambdaW + (W - Wplus);
            LambdaD = LambdaD + (D - Dplus);
            LambdaH = LambdaH + (H - Hplus);
            LambdaX = LambdaX + (X - W*D*H');
            
            loss(i) = cost(V,W,D,H);
        end
        W = Wplus;
        H = Hplus;
        D = Dplus;
    case 'pgrad_euclidean'
        [W,d,H] = initialize(V,k);
        W = W*d;
        for i = 1:maxiter
            W = euclidean_uW(W, V,1,H);
            H = euclidean_uH(H, V,W,1);
            loss(i) = cost(V,W,1,H); 
        end
    case 'pgrad_manifold'
        [W,D,H] = initialize(V,k);
        for i = 1:maxiter
            W = manifold_uW(W, V,D,H);        
            H = manifold_uH(H, V,W,D);
            D = manifold_uS(D, V,W,H);
            loss(i) = cost(V,W,D,H);
        end
end
if ~exist('D')
    D = eye(k);
end
end
function [W,D,H] = initialize(V,k)
    [W,D,H] = svds(V,k);
%     W = randn(size(W));
%     D = diag(diag(randn(size(D))));
%     H = randn(size(H));
    W = max(W,0);
    D = max(D,0);
    H = max(H,0);
end
%% Projected Gradient Variants code
function w_hat = euclidean_uW(WW, VV,SS,HH)
    [dd, nn] = size(WW);
    problem.M = euclideannonnegfactory(dd, nn);

    problem.cost = @(X) cost(VV,X,SS,HH);

    problem.egrad = @(X) -(VV-X*SS*HH')*HH*SS';
    opts.tolgradnorm = 1e-4;
    opts.verbosity = 0;
    opts.linesearch = @linesearch_adaptive;
    w_hat = steepestdescent(problem, WW, opts); 
end
function v_hat = euclidean_uH(HH, VV,WW,SS)
    [dd, nn] = size(HH);
    problem.M = euclideannonnegfactory(dd, nn);

    problem.cost = @(X) cost(VV,WW,SS,X);

    problem.egrad = @(X) -(VV'-X*SS'*WW')*WW*SS;
    opts.tolgradnorm = 1e-4;
    opts.verbosity = 0;
    opts.linesearch = @linesearch_adaptive;
    v_hat = steepestdescent(problem, HH, opts); 
end
%%
function c = cost(VV,WW,SS,HH)
    c = 0.5 * norm(VV-WW*SS*HH', 'fro').^2;
end
%%
function w_hat = manifold_uW(WW, VV,SS,HH)
        [dd, nn] = size(WW);
        problem.M = obliquenonnegfactory(dd, nn);
        
        problem.cost = @(X) cost(VV,X,SS,HH);
        
        problem.egrad = @(X) -(VV-X*SS*HH')*HH*SS';
        opts.tolgradnorm = 1e-4;
        opts.verbosity = 0;
        opts.linesearch = @linesearch_adaptive;
        w_hat = steepestdescent(problem, WW, opts); 
end
function v_hat = manifold_uH(HH, VV,WW,SS)
    [dd, nn] = size(HH);
    problem.M = obliquenonnegfactory(dd, nn);

    problem.cost = @(X) cost(VV,WW,SS,X);

    problem.egrad = @(X) -(VV'-X*SS'*WW')*WW*SS;
    opts.tolgradnorm = 1e-4;
    opts.verbosity = 0;
    opts.linesearch = @linesearch_adaptive;
    v_hat = steepestdescent(problem, HH, opts); 
end

function s_hat = manifold_uS(SS, VV, WW, HH)
    if size(SS,1) ~=  size(SS,2)
        error('Scaling should be supplied as a matrix')
    end
    opts = optimoptions('fmincon');
%         opts.Display = 'iter';
    opts.TolCon = 1e-12;
    opts.Display = 'none';
    opts.DerivativeCheck = 'on';
    opts.Diagnostics = 'off';
    opts.TolFun= 1e-12;
    opts.TolCon = 1e-12;
    SS = diag(SS);
    fun = @(x) costS(x,VV,WW,HH);
    s_hat = fmincon(fun,SS,-eye(3), zeros(3,1), [], [], [], [], [], opts);
    s_hat = diag(s_hat);
end
function [f,g] = costS(s,V,W,H)
    X = diag(s);
    f = cost(V,W,X,H);
    if nargout > 1 % gradient required
        g = diag(W' * (V - W*X*H')*H) ;
    end
end
function optimizer = optimizer_factory(V,k, rho)
    [d,n] = size(V);
    
    optimizer.W.problem.M = obliquefactory(d,k);
    
    optimizer.W.cost = @(X,W,D,H,Wplus,LambdaW)...
        ( cost(X, W, D, H) + rho*norm(W - Wplus + LambdaW,'fro').^2);
    optimizer.W.egrad =@(X,W,D,H,Wplus,LambdaW)...
        (rho*(W-Wplus+LambdaW) - (X - W*D*H')*H*D');
%     optimizer.W.problem.cost = @(W) optimizer.W.cost(X,W,D,H,Wplus,LambdaW);
    
    optimizer.H.problem.M = obliquefactory(n,k);
    optimizer.H.cost = @(X,W,D,H,Hplus,LambdaH)...
        ( cost(X, W, D, H) + rho*norm(H - Hplus + LambdaH,'fro').^2);
    optimizer.H.egrad =@(X,W,D,H,Hplus,LambdaH)...
        (rho*(H-Hplus+LambdaH) - (X'  - H*D'*W')*W*D);
%     optimizer.H.problem.cost = @(H) optimizer.H.cost(X,W,D,H,Hplus,LambdaH);
    optimizer.D.problem.M = euclideanfactory(k,k);
    optimizer.D.cost = @(X,W,D,H,Dplus,LambdaD)...
        ( cost(X, W, D, H) + rho*norm(D - Dplus + LambdaD,'fro').^2);
    optimizer.D.egrad = @(X,W,D,H,Dplus,LambdaD)...
        diag(diag(((D+LambdaD - Dplus) - W'*(X-W*D*H')*H)));
%     optimizer.H.problem.cost = @(H) optimizer.H.cost(X,W,D,H,Hplus,LambdaH);
    optimizer.opts.tolgradnorm = 1e-4;
    optimizer.opts.verbosity = 0;
    optimizer.opts.linesearch = @linesearch_adaptive;
    
    
end