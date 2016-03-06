function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta



% COST

%for thetaTransx for each observation x;
thetaTransx = X * theta;
G = sigmoid(thetaTransx);

%split the summation in the cost fun
% log already arrayfun's
pJ = y' * log(G);
nJ = (1 - y)' * log(1- G);

%now with regularisation term!
thetaJtoM =  [0; theta([2:size(theta,1)],:)];
sumSqthetaJtoM = thetaJtoM' * thetaJtoM;

J = ((-1 / m) * (pJ + nJ)) + ((lambda/(2*m)) * sumSqthetaJtoM);


% GRADIENTS

%diff obs. - actual
D = G - y;
%gradients + regularisation term (0 for first coeff.)
grad = ((1 /m) * (X' * D)) + ((lambda/m) * thetaJtoM);



% =============================================================

end
