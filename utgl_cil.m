function [Y_star, iter, obj_values] = utgl_cil(Ln, Hn, Y, alpha, beta)

mu = 1e-4;
mu_max = 1e6;
rho = 1.2;
iter = 0;
tol = 1e-6;
maxIter = 100;
obj_values = zeros(1, maxIter);

max_iter2 = 3;
fobj2 = zeros(max_iter2, 1);

nv = length(Hn);
[num, k] = size(Hn{1});

R = zeros(num, num, nv);
G = zeros(num, num, nv);
T = zeros(num, num, nv);
P = zeros(k, k, nv);

LL = zeros(num, num, nv);

%Initialization
Hn_updated = zeros(num, k, nv);
for nv_idx = 1 : nv
   Hn_updated(:, :, nv_idx) = Hn{nv_idx};
   P(:, :, nv_idx) = eye(k);
end
max_eigen_values = zeros(1, nv);

while iter < maxIter
    iter = iter + 1;

    % update F
    Ht = zeros(num, k);
    for nv_idx = 1 : nv
        Ht = Ht + Hn_updated(:, :, nv_idx) * P(:, :, nv_idx);    
    end
    Y = update_Y(Y, Ht, num, k);
    Y_star = Y * (Y' * Y + eps * eye(k))^(-0.5);

    % update Pv
    for nv_idx = 1 : nv
        M = Hn_updated(:, :, nv_idx)' * Y_star;
        [Um, ~, Vm] = svd(M, 'econ');
        P(:, :, nv_idx) = Um * Vm';
    end
    
    % update Hv
    A = G - 1 / mu * R;
    for nv_idx = 1 : nv
        iter2 = 0;
        L_star = alpha * Ln{nv_idx} - mu / 2 * (A(:, :, nv_idx) + A(:, :, nv_idx)');

        %cache used
        result = max(max(abs(LL(:, :, nv_idx)-L_star)));
        if result < 0.01
            max_eigen = max_eigen_values(nv_idx);
        else  
            if num > 5000
                [~, d] = eig(L_star);
                d = diag(d);
                [d1, ~] = sort(d,'descend');
                max_eigen = d1(1);
            else
                max_eigen = eigs(L_star, 1 ,'la');
            end    
            max_eigen_values(nv_idx) = max_eigen;
        end
        LL(:, :, nv_idx) = L_star;

        %cache abandoned
%         max_eigen = eigs(L_star, 1 ,'la');

        while iter2 < max_iter2
            iter2 = iter2 + 1;   
            M = (max_eigen * eye(size(L_star))-L_star) * Hn_updated(:, :, nv_idx) + beta * Y_star * P(:, : , nv_idx)';
            [Um, ~, Vm] = svd(M, 'econ');
            Hn_updated(:, :, nv_idx) = Um * Vm';
            fobj2(iter2) = trace(Hn_updated(:, :, nv_idx)' * M);
            if iter2 > 1
                r = (fobj2(iter2)-fobj2(iter2-1))/fobj2(iter2);
%                 disp([iter2, fobj2(iter2), r]);  
                if r < 0.1
                    break;
                end
%             else
%                 disp([iter2, fobj2(iter2)]); 
            end          
        end        
    end

    for nv_idx = 1 : nv
        T( : , : , nv_idx) = Hn_updated(:, :, nv_idx) * Hn_updated(:, :, nv_idx)';
    end
    tempT1 = T + 1 / mu * R;
    tempT = shiftdim(tempT1, 1);
    [tempG, ~, ~] = prox_tnn(tempT, 1 / mu);
    G = shiftdim(tempG, 2);        


     %update R and rho
    R = R + mu * (T - G);    
    mu = min(mu * rho, mu_max);
        
    %calculate the error
    err = max(max(max(abs(G - T))));
    obj_values(1, iter) = err;
%     disp([iter, err]);
    if err < tol
%         disp([iter, err]);
        break;
    end

end

end

