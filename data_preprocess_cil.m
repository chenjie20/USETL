function [L_views, H_views, Z_views, Y] = data_preprocess_cil(data_views, Mn, kn, num_clusters)
% data_views{1}: each column of data_views{1} represents a sample


    num_sample = size(data_views{1}, 2);
    nv = size(data_views, 2);
    L_views = cell(1, nv);
    H_views = cell(1, nv);
    Z_views = cell(1, nv);
    Hs = zeros(num_sample, num_clusters);

    for nv_idx = 1 : nv
        Z = zeros(num_sample, num_sample);
        if nv_idx <= size(Mn, 2)
            %missing_ratio > 0
            cols = abs(Mn{nv_idx} - 1) < 1e-6;
            if  length(find(cols > 0)) < num_sample
                X = data_views{nv_idx}(:, cols);
                W = constructW_PKN(X, kn);
                Z(cols, cols) = W;
            else
                %missing_ratio = 0
                Z = constructW_PKN(data_views{nv_idx}, kn);
            end
        else
            % concatenation
            Z = constructW_PKN(data_views{nv_idx}, kn);
        end
	    Z_views{nv_idx} = Z;

        D = diag(1./sqrt(sum(Z, 2)+ eps));
        W = D * Z * D;
        L_views{nv_idx} = eye(num_sample) - W;

        [U, ~, ~] = svd(W);
        V = U(:, 1 : num_clusters);       
        VV = normr(V);
        H_views{nv_idx} = VV;

        Hs = Hs + VV;

    end

    rand('state', 1000);
    labels = kmeans(Hs, num_clusters, 'maxiter', 1000, 'replicates', 20, 'emptyaction', 'singleton');
    A = sparse(1:num_sample, labels, 1);
    Y = full(A);

    