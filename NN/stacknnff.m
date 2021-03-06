function dsn = stacknnff(dsn,nn, x, opts)
%NNFF performs a feedforward pass
% nn = nnff(nn, x, y) returns an neural network structure with updated
% layer activations, error and loss (nn.a, nn.e and nn.L)

    n = nn.n;
    m = size(x, 1);
    dsn.nn{1}.a{1} = x;
    %% 1.先前向计算第一个模块
    for i = 2 : n-1 
        dsn.nn{1}.a{i-1} = [ones(size(dsn.nn{1}.a{i-1},1),1) dsn.nn{1}.a{i-1}];       
            switch nn.activation_function 
                case 'sigm'
                    % Calculate the unit's outputs (including the bias term)
                    dsn.nn{1}.a{i} = sigm(dsn.nn{1}.a{i - 1} * dsn.nn{1}.W{i - 1}');
                case 'tanh_opt'
                    dsn.nn{1}.a{i} = tanh_opt(dsn.nn{1}.a{i - 1} * dsn.nn{1}.W{i - 1}');
            end

            %dropout
            if(nn.dropoutFraction > 0)
                if(nn.testing)
                    dsn.nn{1}.a{i} = dsn.nn{1}.a{i}.*(1 - nn.dropoutFraction);
                else
                    nn.dropOutMask{i} = (rand(size(nn.a{i}))>nn.dropoutFraction);
                    dsn.nn{1}.a{i} = dsn.nn{1}.a{i}.*nn.dropOutMask{i};
                end
            end
    end  
    dsn.nn{1}.a{n-1} = [ones(size(dsn.nn{1}.a{n-1},1),1) dsn.nn{1}.a{n-1}];
    switch nn.output 
            case 'sigm'
                dsn.nn{1}.a{n} = sigm(dsn.nn{1}.a{n - 1} * dsn.nn{1}.W{n - 1}');
            case 'linear'
                dsn.nn{1}.a{n} = dsn.nn{1}.a{n - 1} * dsn.nn{1}.W{n - 1}';
            case 'softmax'
                dsn.nn{1}.a{n} = dsn.nn{1}.a{n - 1} * dsn.nn{1}.W{n - 1}';
                dsn.nn{1}.a{n} = exp(bsxfun(@minus, dsn.nn{1}.a{n}, max(dsn.nn{1}.a{n},[],2)));
                dsn.nn{1}.a{n} = bsxfun(@rdivide, dsn.nn{1}.a{n}, sum(dsn.nn{1}.a{n}, 2)); 
    end   
    output = abs(dsn.nn{1}.a{n}); %output用于记录前面所有模块的输出
%% 2.再计算后续模块

    for j = 2 : opts.stack
       %Get a{n} for the jth layer, then stack a{n} as the new input of
        %a{1} rather than just using x
        dsn.nn{j}.a{1} = [ones(m,1) output x]; %从第二个模块起，每个模块的输入为前面所有模块的输出加上原始输入 
        %feedforward pass
        for i = 2 : n-1
            switch nn.activation_function 
                case 'sigm'
                    % Calculate the unit's outputs (including the bias term)
                    dsn.nn{j}.a{i} = sigm(dsn.nn{j}.a{i - 1} * dsn.nn{j}.W{i - 1}');
                case 'tanh_opt'
                    dsn.nn{j}.a{i} = tanh_opt(dsn.nn{j}.a{i - 1} * dsn.nn{j}.W{i - 1}');
            end

            %dropout
            if(nn.dropoutFraction > 0)
                if(nn.testing)
                    dsn.nn{j}.a{i} = dsn.nn{j}.a{i}.*(1 - nn.dropoutFraction);
                else
                    nn.dropOutMask{i} = (rand(size(nn.a{i}))>nn.dropoutFraction);
                    dsn.nn{j}.a{i} = dsn.nn{j}.a{i}.*nn.dropOutMask{i};
                end
            end

            %calculate running exponential activations for use with sparsity
            if(nn.nonSparsityPenalty>0)
                nn.p{i} = 0.99 * nn.p{i} + 0.01 * mean(dsn.nn{j}.a{i}, 1);
            end

            %Add the bias term
            
        end
        dsn.nn{j}.a{n-1} = [ones(size(dsn.nn{j}.a{n-1},1),1) dsn.nn{j}.a{n-1}];
        switch nn.output 
            case 'sigm'
                dsn.nn{j}.a{n} = sigm(dsn.nn{j}.a{n - 1} * dsn.nn{j}.W{n - 1}');
            case 'linear'
                dsn.nn{j}.a{n} = dsn.nn{j}.a{n - 1} * dsn.nn{j}.W{n - 1}';
            case 'softmax'
                dsn.nn{j}.a{n} = dsn.nn{j}.a{n - 1} * dsn.nn{j}.W{n - 1}';
                dsn.nn{j}.a{n} = exp(bsxfun(@minus, dsn.nn{j}.a{n}, max(dsn.nn{j}.a{n},[],2)));
                dsn.nn{j}.a{n} = bsxfun(@rdivide, dsn.nn{j}.a{n}, sum(dsn.nn{j}.a{n}, 2)); 
        end  
        output = [output abs(dsn.nn{j}.a{n})];
    end
end
