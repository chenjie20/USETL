function Y = update_Y(Y,G, num_sample, num_cluster)
% C. Tang, Z. Li, J. Wang, X. Liu, W. Zhang, and E. Zhu, 
% "Unified one-step multi-view spectral clustering," IEEE Trans. Knowl. Data Eng., pp.1¨C11, May 2022.
yg = diag(Y'* G)';
yy = diag(Y'*Y+eps*eye(num_cluster))';
for i = 1 : num_sample
    gi = G(i,:);
    yi = Y(i,:);
    si = (yg+gi.*(1-yi))./sqrt(yy+1-yi) - (yg-gi.*yi)./sqrt(yy-yi);
    [~,index] = max(si(:));
    Y(i,:) = 0;
    Y(i,index) = 1;
end
end

