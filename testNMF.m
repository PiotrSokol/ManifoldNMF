function []= testNMF()

k = 3;
Wg = normalize_columns(max(randn(8,k), 0) );
Hg = normalize_columns(max(randn(100,k), 0) );
Dg = diag( exprnd(100, [k,1] ) );
V = Wg*Dg*Hg';
V = V+randn(size(V))*20;

W = randn(size(Wg));
H = randn(size(Hg'));
% [W, H, loss_comp, time] = nmf_kl_admm(V, W, H, 2, 1, 500);
% loss_comp = loss_comp./(norm(V,'fro').^2);

[W,D,H, loss_madmm] = nmf(V,3, 'admm_manifold');
loss_madmm = loss_madmm./(norm(V,'fro').^2);

[W1,D1,H1 loss_eadmm] = nmf(V,3, 'admm_euclidean');
loss_eadmm = loss_eadmm./(norm(V,'fro').^2);

[~,~,~, loss_epgrad] = nmf(V,3, 'pgrad_euclidean');
loss_epgrad = loss_epgrad./(norm(V,'fro').^2);

[~,~,~, loss_mpgrad] = nmf(V,3, 'pgrad_manifold');
loss_mpgrad = loss_mpgrad./(norm(V,'fro').^2);






semilogy( loss_madmm );
hold on;
semilogy( loss_eadmm );
semilogy(loss_epgrad);
semilogy(loss_mpgrad)
legend({'MADMM', 'ADMM','PGRAD','M PGRAD'})
    function X = normalize_columns(X)
        % This is faster than norms(X, 2, 1) for small X, and as fast for large X.
        maxs = max(X,[],1);
        maxs(maxs>0) = Inf;
        X(bsxfun(@eq, X, maxs)) = 1;
        X = max(X,0);
        nrms = sqrt(sum(X.^2, 1));
        X = bsxfun(@times, X, 1./nrms);
    end
end