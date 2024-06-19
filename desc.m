% 读取CSV文件，注意要将文件路径替换为实际路径
data = dlmread('iris.csv', ',', 1, 0);

% 分离特征和标签
X = data(:, 1:4);
y = data(:, 5);

% 将标签转换为分类编号
labels = unique(y);
y_numeric = zeros(size(y));
for i = 1:length(labels)
    y_numeric(y == labels(i)) = i;
end

% 将数据随机分为训练集和测试集
[m, n] = size(X);
rand_indices = randperm(m);
train_ratio = 0.7;
train_size = round(train_ratio * m);
X_train = X(rand_indices(1:train_size), :);
y_train = y_numeric(rand_indices(1:train_size), :);
X_test = X(rand_indices(train_size+1:end), :);
y_test = y_numeric(rand_indices(train_size+1:end), :);

% 将数据两两配对，在二维展示图像
colors = ['g', 'r', 'b']; % 用于不同类别的颜色

figure;
for i = 1:n
    for j = 1:n
        subplot(n, n, (i-1)*n + j);
        hold on;
        if i == j
            % 在对角线的子图中绘制直方图
            for k = 1:length(labels)
                [counts, binCenters] = hist(X(y_numeric == k, i), 10);
                bar(binCenters, counts, 'FaceColor', colors(k), 'EdgeColor', colors(k));
            end
        else
            % 在其他子图中绘制散点图
            for k = 1:length(labels)
                scatter(X(y_numeric == k, j), X(y_numeric == k, i), [], colors(k), 'filled');
            end
        end
        if i == n
            xlabel(['Feature ' num2str(j)]);
        end
        if j == 1
            ylabel(['Feature ' num2str(i)]);
        end
        hold off;
    end
end

legend(arrayfun(@(x) ['Class ' num2str(x)], labels, 'UniformOutput', false), 'Location', 'northeastoutside');


