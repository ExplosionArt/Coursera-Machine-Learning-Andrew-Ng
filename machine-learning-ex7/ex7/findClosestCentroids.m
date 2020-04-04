function idx = findClosestCentroids(X, centroids)
%FINDCLOSESTCENTROIDS computes the centroid memberships for every example
%   idx = FINDCLOSESTCENTROIDS (X, centroids) returns the closest centroids
%   in idx for a dataset X where each row is a single example. idx = m x 1 
%   vector of centroid assignments (i.e. each entry in range [1..K])
%

% Set K
%In our example, K is set as 3 i.e we have to compute 3 centroids
K = size(centroids, 1);

% You need to return the following variables correctly.
%Our X has 300 training samples, so size of idx has to be 300 x 1
idx = zeros(size(X,1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Go over every example, find its closest centroid, and store
%               the index inside idx at the appropriate location.
%               Concretely, idx(i) should contain the index of the centroid
%               closest to example i. Hence, it should be a value in the 
%               range 1..K
%
% Note: You can use a for-loop over the examples to compute this.
%

%Run a loop till all training examples i.e m
for i = 1:size(X,1)
    %Choose a large random value
    min_distance = 10^6;
    %Now for each training sample, check for all centroids
    for j = 1:K
        %Check if jth centroid has less distance than min_distance
        %Accordingly, we have to reduce sum of norm of x(i) - centroid assigned to i
        temp_distance = sum(norm((X(i,:)-centroids(j,:))).^2);
        if temp_distance < min_distance
            %Here? That means jth centroid is near to ith training sample
            min_distance = temp_distance;
            idx(i) = j;
        end
    end
end
% =============================================================

end