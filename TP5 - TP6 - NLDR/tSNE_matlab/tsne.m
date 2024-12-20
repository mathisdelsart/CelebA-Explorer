function ydata = tsne(X, no_dims, perplexity)
%TSNE Performs symmetric t-SNE on dataset X
%
%   mappedX = tsne(X, no_dims, perplexity)
%
% The function performs symmetric t-SNE on the NxD dataset X to reduce its 
% dimensionality to no_dims dimensions (default = 2). 
% The perplexity of the Gaussian kernel that is employed 
% can be specified through perplexity (default = 30). 
% The low-dimensional data representation is returned in mappedX.
%
%
% (C) Laurens van der Maaten, 2010
% University of California, San Diego


if ~exist('no_dims', 'var') || isempty(no_dims)
    no_dims = 2;
end
if ~exist('perplexity', 'var') || isempty(perplexity)
    perplexity = 30;
end

% Compute pairwise distance matrix
sum_X = sum(X .^ 2, 2);
D = bsxfun(@plus, sum_X, bsxfun(@plus, sum_X', -2 * (X * X')));

% Compute joint probabilities
P = d2p(D, perplexity, 1e-5);                                           % compute affinities using fixed perplexity
clear D

% Run t-SNE
ydata = tsne_p(P, [], no_dims);
end
    