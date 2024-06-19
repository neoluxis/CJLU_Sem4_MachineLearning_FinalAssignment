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

% 初始化神经网络参数
input_layer_size = 4;   % 输入层特征数
hidden_layer_size = 10; % 隐藏层单元数
num_labels = length(labels); % 分类数

% 随机初始化权重
epsilon_init = 0.12;
Theta1 = rand(hidden_layer_size, input_layer_size + 1) * 2 * epsilon_init - epsilon_init;
Theta2 = rand(num_labels, hidden_layer_size + 1) * 2 * epsilon_init - epsilon_init;

% Sigmoid激活函数
function g = sigmoid(z)
    g = 1.0 ./ (1.0 + exp(-z));
end

% Sigmoid梯度函数
function g = sigmoidGradient(z)
    g = sigmoid(z) .* (1 - sigmoid(z));
end

% 前向传播
function [a1, z2, a2, z3, a3] = forward_propagate(X, Theta1, Theta2)
    m = size(X, 1);
    a1 = [ones(m, 1) X]; % 添加偏置项
    z2 = a1 * Theta1';
    a2 = [ones(m, 1) sigmoid(z2)];
    z3 = a2 * Theta2';
    a3 = sigmoid(z3);
end

% 代价函数
function J = cost_function(X, y, Theta1, Theta2, num_labels)
    m = size(X, 1);
    [~, ~, ~, ~, h] = forward_propagate(X, Theta1, Theta2);
    y_matrix = eye(num_labels)(y,:);
    J = (1/m) * sum(sum(-y_matrix .* log(h) - (1 - y_matrix) .* log(1 - h)));
end

% 反向传播更新权重
function [Theta1, Theta2] = backpropagate(X, y, Theta1, Theta2, alpha, num_labels)
    m = size(X, 1);
    y_matrix = eye(num_labels)(y,:);

    for i = 1:m
        % 前向传播
        a1 = [1; X(i,:)'];
        z2 = Theta1 * a1;
        a2 = [1; sigmoid(z2)];
        z3 = Theta2 * a2;
        a3 = sigmoid(z3);

        % 计算误差
        delta3 = a3 - y_matrix(i,:)';
        delta2 = (Theta2' * delta3)(2:end) .* sigmoidGradient(z2);

        % 累积梯度
        Theta1_grad = delta2 * a1';
        Theta2_grad = delta3 * a2';

        % 更新权重
        Theta1 = Theta1 - alpha * Theta1_grad;
        Theta2 = Theta2 - alpha * Theta2_grad;
    end
end

% 训练神经网络
alpha = 0.1; % 学习率
num_iters = 1000; % 迭代次数
for i = 1:num_iters
    [Theta1, Theta2] = backpropagate(X_train, y_train, Theta1, Theta2, alpha, num_labels);
end

% 预测
function p = predict(Theta1, Theta2, X)
    [~, ~, ~, ~, h] = forward_propagate(X, Theta1, Theta2);
    [~, p] = max(h, [], 2);
end

% 评估模型
predictions = predict(Theta1, Theta2, X_test);
accuracy = mean(double(predictions == y_test)) * 100;
fprintf('BP神经网络分类准确率: %.2f%%\n', accuracy);

% 训练朴素贝叶斯分类器
function model = train_naive_bayes(X, y)
    labels = unique(y);
    num_labels = length(labels);
    [m, n] = size(X);
    model.mean = zeros(num_labels, n);
    model.variance = zeros(num_labels, n);
    model.priors = zeros(num_labels, 1);

    for i = 1:num_labels
        X_i = X(y == labels(i), :);
        model.mean(i, :) = mean(X_i);
        model.variance(i, :) = var(X_i);
        model.priors(i) = size(X_i, 1) / m;
    end
end

% 预测朴素贝叶斯分类器
function p = predict_naive_bayes(model, X)
    num_labels = length(model.priors);
    [m, n] = size(X);
    p = zeros(m, 1);

    for i = 1:m
        probabilities = zeros(num_labels, 1);
        for j = 1:num_labels
            likelihood = -0.5 * sum(log(2 * pi * model.variance(j, :))) - 0.5 * sum(((X(i, :) - model.mean(j, :)) .^ 2) ./ model.variance(j, :));
            probabilities(j) = likelihood + log(model.priors(j));
        end
        [~, p(i)] = max(probabilities);
    end
end

% 训练模型
model = train_naive_bayes(X_train, y_train);

% 评估模型
predictions = predict_naive_bayes(model, X_test);
accuracy = mean(double(predictions == y_test)) * 100;
fprintf('朴素贝叶斯分类准确率: %.2f%%\n', accuracy);



