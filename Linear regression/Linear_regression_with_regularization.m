% the dataset used is: 
% Tsanas,Athanasios and Xifara,Angeliki. (2012). 
% Energy Efficiency. UCI Machine Learning 
% Repository. https://doi.org/10.24432/C51307.
%
% the dataset has:
% - 8 features, integers,Real
% - 768 instances
% - heating load is the only target in the dataset, others are features

clc; close all; clear all;

% load the dataset
dataset = readtable("energy_efficiency_data_heating_load.csv");
data = table2array(dataset);
% visualize the features and checking for outliers
figure('Name',"Dataset features", "NumberTitle","off") 
tiledlayout(4,2,"TileSpacing","compact", "Padding","compact")
for i = 1:8
    nexttile
    feature = data(:,i);
    x = linspace(-10,100,height(feature));
    scatter(x,feature)
    grid on
end

% splitting the dataset into training and validation
validation_count = floor(height(data(:,1)) * 0.35);
training_count = height(data(:,1)) - validation_count;

train_set = data(1:training_count,1:8);
val_set = data(training_count+1:end, 1:8);
train_y = data(1:training_count,9);
val_y = data(training_count+1:end,9);

% Running the normalization function for the training and validation sets
[train_normalized, mu, sigma] = normalizeFeatures(train_set);
[val_normalized, mu, sigma] = normalizeFeatures(val_set);

% set the learning rate and number of iterations and lambda
alpha = 0.00001;          % learning rate
iter = 100000;         % number of iterations
lambda = 1;            % regularization parameter

% adding the intercept column
training_set = [ones(height(train_normalized),1) train_normalized];
validation_set = [ones(height(val_normalized),1) val_normalized];
features = width(training_set);

% implementing the gradient descent with regularization
m = height(train_y);                  % number of training instances
J_history = zeros(iter, 1);           % training cost
val_history = zeros(iter, 1);         % validation cost
theta = ones(features, 1);            % parameters

for i = 1:iter
    J_history(i) = 1/(2*m)*sum((training_set*theta-train_y).^2) + lambda/(2*m)*sum(theta.^2);       % calculate the training cost
    val_history(i) = 1/(2*m)*sum(((validation_set*theta)-val_y).^2) + lambda/(2*m)*sum(theta.^2);   % calculate the validation cost
    grad = (1/m)*training_set'*((training_set*theta)-train_y) + ((lambda/m) * theta);               % the gradient of the cost function
    theta = theta - alpha*grad;                                                                     % updating the learning parameters
end

train_cost = J_history(end,end)
val_cost = val_history(end,end)


% Plot the training and validation costs
x = linspace(10000,iter, iter-1000000);
figure('Name',"Training and validation costs", "NumberTitle","off") 
plot(x,J_history(1000001:end))
hold on 
plot(x,val_history(1000001:end))
title("Training and validation")
legend('Training cost', 'Validation cost', 'Location','northeast')
xlabel('Iterations')
ylabel('Cost')
grid on

% implementing a pre-processing normalization function
function [normalized, mu, sigma] = normalizeFeatures(x)
    
    feature_num = width(x);
    row_num = height(x);
    normalized = zeros(row_num, feature_num);
    mu = zeros(1, feature_num);
    sigma = zeros(1,feature_num);

    for feature = 1:feature_num

        Avg = mean(x(:, feature));
        normalized(:, feature) = (x(:, feature)-Avg) / (max(x(:,feature))-min(x(:,feature)));
        mu(1, feature) = Avg;
        sigma(1, feature) = std(x(:,feature));
    end
end


