function [J] = computeCost (X, y, theta)
m = length(y); % number of training examples
J = 0;

predictions = X * theta;  % predictions of hypothesis on all m examples

J = 1/(2*m)*(predictions - y)'*(predictions - y);
endfunction
