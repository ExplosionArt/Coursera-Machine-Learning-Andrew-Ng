function centroids = computeCentroids(X, idx, K)
%COMPUTECENTROIDS returns the new centroids by computing the means of the 
%data points assigned to each centroid.
%   centroids = COMPUTECENTROIDS(X, idx, K) returns the new centroids by 
%   computing the means of the data points assigned to each centroid. It is
%   given a dataset X where each row is a single data point, a vector
%   idx of centroid assignments (i.e. each entry in range [1..K]) for each
%   example, and K, the number of centroids. You should return a matrix
%   centroids, where each row of centroids is the mean of the data points
%   assigned to it.
%

% Useful variables
[m n] = size(X);

% You need to return the following variables correctly.
centroids = zeros(K, n);


% ====================== YOUR CODE HERE ======================
% Instructions: Go over every centroid and compute mean of all points that
%               belong to it. Concretely, the row vector centroids(i, :)
%               should contain the mean of the data points assigned to
%               centroid i.
%
% Note: You can use a for-loop over the centroids to compute this.
%

%We know that there are m training examples i.e row of X or size(X,1)
m = size(X,1);

%Firstly, The outer loop will run for all K centroids
for i = 1:K
    %Now for each training sample, check if closest centroid is i or not. If yes, sum it and increment a count
    temp_count = 0;
    for j = 1:m
        %Check if closest centroid of j is i
        if idx(j,:) == i
            %If yes, add the point to that centroid computation
            centroids(i,:) = (centroids(i,:) + X(j,:));
            temp_count = temp_count + 1;
        end
    end
    centroids(i,:) = centroids(i,:)/temp_count;
end
% =============================================================


end

