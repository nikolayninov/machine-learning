function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 0;
sigma = 0;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%
best_perc = 0;
for loc_c = [0.01,0.03,0.1,0.3,1,3,10,30] 
  for loc_sigma = [0.01,0.03,0.1,0.3,1,3,10,30]
    loc_model = svmTrain(X, y, loc_c, @(x1, x2) gaussianKernel(x1, x2, loc_sigma));
    loc_pred = svmPredict(loc_model,Xval);
    loc_perc = sum(loc_pred==yval)/size(y)(1);
    if (loc_perc > best_perc)
      best_perc = loc_perc;
      C = loc_c;
      sigma = loc_sigma;
    endif
  endfor
endfor





% =========================================================================

end
