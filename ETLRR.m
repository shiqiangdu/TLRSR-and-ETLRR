function [ L,E,N,J_rank,err] =ETLRR(X,D,max_iter,DEBUG,alpha,beta,Enorm)
%%% 20200901
[n1,n2,n3]=size(X);
[~,n4,~]=size(D);

%% E=Y1 n1 n2 n3
E = zeros(n1,n2,n3);
Y1=E;
N=E;

%% L=J=Y2 n4 n2 n3
L=zeros(n4,n2,n3);
J=L;
Y2=L;

mu = 1e-4;
max_mu = 1e+8;
tol = 1e-8;
rho = 1.2;
iter = 0;
% max_iter = 500;

%%?Pre compute
Din = t_inverse(D);
DT = tran(D);
while iter < max_iter
    iter = iter+1;
    
    %% update Jk
    J_pre = J;
    P = L-Y2/mu;
    [J,J_nuc,J_rank,~] = prox_tnn(P,1/mu);
    
    %% update Lk
    L_pre=L;
    R2=J+Y2/mu;
    R1=X-E+Y1/mu-N;
    L=tprod(Din, R2+tprod(DT,R1));
    
    %% update Ek
    E_pre = E;
    T=X-tprod(D,L)+Y1/mu;
    R3=T-N;
    E = prox_l1( R3, alpha/mu );
    
    %% update Nk
    N_pre=N;
    Q=T-E;
    switch Enorm
        
        %%% 从张量侧向转化为前向，然后转为以矩阵进行2,1 范数优化
        case 21
            Qtran=permute(Q,[1,3,2]);
            Qm=zeros(n1*n3,n2);
            for k=1:n2
                Qk=Qtran(:,:,k);
                Qm(:,k)=Qk(:);
            end
            QmE=L21_solver(Qm,beta/mu);
            Et=zeros(n1,n3,n2);
            for k=1:n2
                Et(:,:,k)=reshape(QmE(:,k),n1,n3);
            end
            N=permute(Et,[1,3,2]);
        case 2
            N=mu/(beta+mu)*Q;
    end
    
    %% check convergence
    leq1 = X-tprod(D,L)-E-N;
    leq2 = J-L;
    leqm1 = max(abs(leq1(:)));
    leqm2 = max(abs(leq2(:)));
    
    difJ = max(abs(J(:)-J_pre(:)));
    difE = max(abs(E(:)-E_pre(:)));
    difL = max(abs(L(:)-L_pre(:)));
    difN = max(abs(N(:)-N_pre(:)));
    err = max([leqm1,leqm2,difJ,difL,difE,difN]);
    if DEBUG && (iter==1 || mod(iter,20)==0)
        sparsity=length(find(E~=0));
        fprintf('iter = %d, obj = %.3f, err = %.8f, beta=%.2f, rankL = %d, sparsity=%d\n'...
            , iter,J_nuc+alpha*norm(E(:),1),err,mu,J_rank,sparsity);
    end
    if err < tol
        break;
    end
    
    %% update Lagrange multiplier and  penalty parameter beta
    Y1 = Y1 + mu*leq1;
    Y2 = Y2 + mu*leq2;
    mu = min(mu*rho,max_mu);
end
end
