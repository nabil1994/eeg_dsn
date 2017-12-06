function [ X ] = sample( P )
% Samples X according to probability P
X = double(P > rand(size(P)));

end

