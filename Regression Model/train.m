function [theta, J_history] = train (X,y,theta,alpha, trainings)
m = length(y);
J_history = zeros(trainings,1);

for i = 1:trainings,
  theta = theta - alpha * (1/m)*sum((X*theta-y).*X)';
  
  J_history(i) = computeCost(X,y,theta);
  
end
endfunction
