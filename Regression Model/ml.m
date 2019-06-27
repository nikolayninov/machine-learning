clear ; close all; clc


data = load('ex1data2.txt');
X = data(:,1:2);
y = data(:,3);
m = length(y);


% Uncomment when you are including new data
fprintf('Normalizing Features...\n');
[X mu sigma] = featureNormalize(X);
save('normalized','X');

X = [ones(m,1),X];

fprintf('Training...\n');

learning_rate = 0.02;
trainings = 10000;

% reset params
%theta_res = zeros(3,1);
%save params.txt theta_res;
theta = [struct2cell(load('params.txt')){:}];
[theta, J_history] = train(X, y, theta, learning_rate, trainings);

% Plotting the convergence graph of J
%plot(1:numel(J_history), J_history, '-b', 'LineWidth', 2);
%xlabel('Number of iterations');
%ylabel('Cost J');

% Display gradient descent's result
fprintf('Theta computed from gradient descent: \n');
fprintf(' %f \n', theta);
fprintf('\n');
theta_normal = normalEqn(X,y);

fprintf('Theta computed from normal equation: \n');
fprintf(' %f \n', theta_normal);
fprintf('\n');

size = 1650;
room_cnt = 3;
price =[1 (([size room_cnt]-mu)./sigma)]*theta;

fprintf('Predicted price of a %f sq-ft, %i br house: %f \n',size, room_cnt,price);
