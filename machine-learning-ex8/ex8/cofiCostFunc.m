function [J, grad] = cofiCostFunc(params, Y, R, num_users, num_movies, ...
                                  num_features, lambda)
%COFICOSTFUNC Collaborative filtering cost function
%   [J, grad] = COFICOSTFUNC(params, Y, R, num_users, num_movies, ...
%   num_features, lambda) returns the cost and gradient for the
%   collaborative filtering problem.
%

% Unfold the U and W matrices from params
X = reshape(params(1:num_movies*num_features), num_movies, num_features);
Theta = reshape(params(num_movies*num_features+1:end), ...
                num_users, num_features);

            
% You need to return the following values correctly
J = 0;
X_grad = zeros(size(X));
Theta_grad = zeros(size(Theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost function and gradient for collaborative
%               filtering. Concretely, you should first implement the cost
%               function (without regularization) and make sure it is
%               matches our costs. After that, you should implement the 
%               gradient and use the checkCostFunction routine to check
%               that the gradient is correct. Finally, you should implement
%               regularization.
%
% Notes: X - num_movies  x num_features matrix of movie features
%        Theta - num_users  x num_features matrix of user features
%        Y - num_movies x num_users matrix of user ratings of movies
%        R - num_movies x num_users matrix, where R(i, j) = 1 if the 
%            i-th movie was rated by the j-th user
%
% You should set the following variables correctly:
%
%        X_grad - num_movies x num_features matrix, containing the 
%                 partial derivatives w.r.t. to each element of X
%        Theta_grad - num_users x num_features matrix, containing the 
%                     partial derivatives w.r.t. to each element of Theta
%

% Okay so the cost function is big, So do it part by part
% Firstly, calculate how much is the distance between predicted values and actual values
% Note the Dimension of X and Theta described above. One of them has to be Transposed. 
prediction_error = (X * Theta' - Y);

% So Now, prediction_error becomes of size num_movies x num_users. However, one thing still remains is that Only error of those terms will be calculated for which users have provided a rating. i.e R comes into play here
% Note that Element wise multiplication would be needed as both are of same dimensions but only those (i,j) for which R(i,j)==1 will be considered.
prediction_error = prediction_error .* R;

% Our basic prediction_error matrix is ready. However, the cost is just a scalar value.So we basically need the sum of all the squared values of the prediction_error matrix
% Since it is a 2D matrix, one summation would convert matrix to a vector and another sum is required to get a single scalar value
sum_error = sum(sum(prediction_error .^ 2));

% Now, a few steps later, pdf mentions to add regularization terms to this cost function
% I think it will be beneficial if I split the two regularization terms, one for X and other for Theta
% Same above summation logic aplies to Theta and X as well
regularization_term_Theta = (lambda / 2) * sum(sum(Theta .^ 2));
regularization_term_X = (lambda / 2) * sum(sum(X .^ 2));

% So finally, our Cost J is 1/2 * error + regTheta + regX
J = 0.5 * sum_error + regularization_term_Theta + regularization_term_X;

% Only gradient calculation now remains for Gradient Descent. Since both X and Theta need to be minimized, X_grad and Theta_grad will be present
% Our prediction error already takes care of the part where Only rated movies are considered. Multiplying that with Theta gives which user has rated which movies and for those movies, what is the prediction error
% No tranpose required as dimensions of error: num_movies x num_users and Theta : num_users x num_features and 
X_grad = prediction_error * Theta + lambda * X;

% Same logic for Theta_Grad except that it needs to be Transposed because error has dim of num_movies x num_users while X : num_movies x num_features
Theta_grad = prediction_error' * X + lambda * Theta;
% =============================================================

grad = [X_grad(:); Theta_grad(:)];

end
