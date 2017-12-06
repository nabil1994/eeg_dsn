function [labels, a] = nnpredict(dsn, nn, x, opt)
    n = numel(dsn.nn);
    nn.testing = 1;
%      nn = nnff(nn, x, zeros(size(x,1), nn.size(end)), opt.stack);
    dsn = stacknnff(dsn, nn, x, opt);
%    nn = nnff(nn, x, y, opt.stack);
    nn.testing = 0;
    
    [~, i] = max(dsn.nn{n}.a{end},[],2);
    labels = i;
    a = dsn.nn{n}.a{end};
end
