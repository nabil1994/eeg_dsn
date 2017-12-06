function [ dsn ] = dsntrain( dsn, x, opts )

    n = numel(dsn.rbm);
    tic;
    dsn.rbm{1} = rbmtrain(dsn.rbm{1}, x, opts);
    for i = 2 : n
        x = rbmup(dsn.rbm{i - 1}, x);
        dsn.rbm{i} = rbmtrain(dsn.rbm{i}, x, opts);
    end
    t = toc
end

