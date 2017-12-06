function [ nn ] = piledhvdsns( nn, dsn )
%PILEDSNS Summary of this function goes here
%   Detailed explanation goes here
nn.size(2) = nn.size(2)+ dsn.sizes(2);
n = nn.n;
for i = 2 : n-1   
    % weights and weight momentum
    nn.W{i-1} = [nn.W{i-1};dsn.rbm{i-1}.c dsn.rbm{i-1}.W]; 
    nn.vW{i - 1} = zeros(size(nn.W{i - 1}));
    % average activations (for use with sparsity)
    nn.p{i} = zeros(1, nn.size(i));
end
nn.W{n-1} = [nn.W{n-1} normrnd(0,0.01,nn.size(n), dsn.sizes(n - 1))]; 
nn.vW{n - 1} = zeros(size(nn.W{n - 1}));
end

