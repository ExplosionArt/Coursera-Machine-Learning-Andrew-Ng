function Z = projectData(X, U, K)
%PROJECTDATA Computes the reduced data representation when projecting only 
%on to the top k eigenvectors
%   Z = projectData(X, U, K) computes the projection of 
%   the normalized inputs X into the reduced dimensional space spanned by
%   the first K columns of U. It returns the projected examples in Z.
%

% You need to return the following variables correctly.
Z = zeros(size(X, 1), K);

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the projection of the data using only the top K 
%               eigenvectors in U (first K columns). 
%               For the i-th example X(i,:), the projection on to the k-th 
%               eigenvector is given as follows:
%                    x = X(i, :)';
%                    projection_k = x' * U(:, k);
%

%As usual, store size(X,1) in m
m = size(X,1);

%Run a loop for all training samples
for i = 1:m
    %First calculate transpose of ith sample as this will be needed further in calculation of Z
    x = X(i,:)';
    %For each centroid
    for k = 1:K
        % z = x' * Ureduce
        projection_k = x' * U(: , k);
        Z(i,k) = projection_k;
    end
end

% =============================================================

end
