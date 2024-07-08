function [data_views] = normalize_multiview_data(data_views)
% data_views{1}: each column of data_views{1} represents a sample

nv = size(data_views, 2);
num_sample = size(data_views{1}, 2);
for i = 1 : nv
    for  j = 1 : num_sample
        std_vaule = std(data_views{i}(:, j));
        if std_vaule < eps
            std_vaule = eps;
        end
        data_views{i}(:, j) = (data_views{i}(:, j) - mean(data_views{i}(:, j)))/std_vaule;
    end    
end