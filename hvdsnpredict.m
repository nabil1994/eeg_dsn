function [labels, a] = hvdsnpredict(nn, x)
    nn.testing = 1;
    nn = nnff(nn, x);

    nn.testing = 0;
    
    [~, i] = max(nn.a{end},[],2);
    labels = i;
    a = nn.a{end};
end
