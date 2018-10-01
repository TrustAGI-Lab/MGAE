function [V E F P R nmi avgent AR C, ACC] = baseline_spectral_onRW(Prob,numClust,truth,projev)
% spectral clustering using transition matrix K. Clustering is done by
% running k-means on the top-'numClust' eigenvectors of the normalized Laplacian
% INPUT:
% K: N x N similarity matrix. 
% numClust: desired number of clusters
% truth: N x 1 vector of ground truth clustering
% projev: number of top eigenvectors to return in V
% OUTPUT:
% V: top-'projev' eigenvectors of the Laplacian
% E: top-'projev' eigenvalues of the Laplacian
% F, P, R, nmi, avgent, AR: F-score, Precision, Recall, normalized mutual
% information, average entropy, Avg Rand Index
% C: obtained cluster labels 

%     [p,~]=eigs(Prob',1);
%     p=p/sum(p);
%     P=(diag(p)*Prob+Prob'*diag(p))/2;
    numEV = numClust*projev;
    opts.disp = 0;
    [V E] = eigs(Prob,ceil(numEV),'lm',opts);  
    U = V(:,1:ceil(numClust*1));
    
    %[U E] = eig(L);   
    %[E1 I] = sort(diag(E));  %sort in increasing order
    %U = U(:,I(end-numEV+1:end));
    if (1)
    norm_mat = repmat(sqrt(sum(U.*U,2)),1,size(U,2));
    %%avoid divide by zero
    for i=1:size(norm_mat,1)
        if (norm_mat(i,1)==0)
            norm_mat(i,:) = 1;
        end
    end
    U = U./norm_mat;
    end
    %fprintf('running k-means...\n');
    
    for i=1:50
        C = kmeans(U,numClust,'EmptyAction','drop');
        [Fi(i),Pi(i),Ri(i)] = compute_f(truth,C);
        [A nmii(i) avgenti(i)] = compute_nmi(truth,C);
        if (min(truth)==0)
            [ARi(i),RIi(i),MIi(i),HIi(i)]=RandIndex(truth+1,C);
        else
            [ARi(i),RIi(i),MIi(i),HIi(i)]=RandIndex(truth,C);
        end
        res = bestMap(truth,C);
        ACCi(i) = length(find(truth == res))/length(truth);       
    end
    F(1) = mean(Fi); F(2) = std(Fi);
    P(1) = mean(Pi); P(2) = std(Pi);
    R(1) = mean(Ri); R(2) = std(Ri);
    ACC(1) = mean(ACCi); ACC(2) = std(ACCi);
    nmi(1) = mean(nmii); nmi(2) = std(nmii);
    avgent(1) = mean(avgenti); avgent(2) = std(avgenti);
    AR(1) = mean(ARi); AR(2) = std(ARi);
    
    %fprintf('F: %f(%f)\n', F(1), std(Fi));
    %fprintf('P: %f(%f)\n', P(1), std(Pi));    
    %fprintf('R: %f(%f)\n', R(1), std(Ri));
%     fprintf('nmi: %f(%f)\n', nmi(1), std(nmii));
    %fprintf('avgent: %f(%f)\n', avgent(1), std(avgenti));
    %fprintf('AR: %f(%f)\n', AR(1), std(ARi));
    
    