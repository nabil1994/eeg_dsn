function [er, bad, a, b] = nntest(dsn, nn, x, y, opt)
    [labels, a] = nnpredict(dsn, nn, x, opt);
    [~, expected] = max(y,[],2);
    bad = find(labels ~= expected);    
    er = numel(bad) / size(x, 1)
    b = expected;
end
