function [er, bad, a, b] = hvdsntest(nn, x, y)
    tic;
    [labels, a] = hvdsnpredict(nn, x);
    toc;
    [~, expected] = max(y,[],2);
    bad = find(labels ~= expected);    
    er = numel(bad) / size(x, 1);
    b = expected;
end
