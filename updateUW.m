function nn = updateUW(nn, x, y, opts)

    n = nn.n;
    m = size(x, 1);
    momentum = opts.momentum;
    alpha = opts.alpha;
    %x = [ones(m,1) x];

    nn.a{1} = x;
%% 1.前向传播 2.更新W 3.更新U
    %feedforward pass
    for i = 2 : n-1
        nn.a{i-1} = [ones(size(nn.a{i-1},1),1) nn.a{i-1}];
        switch nn.activation_function 
            case 'sigm'
                % Calculate the unit's outputs (including the bias term)
                nn.a{i} = sigm(nn.a{i - 1} * nn.W{i - 1}');
            case 'tanh_opt'
                nn.a{i} = tanh_opt(nn.a{i - 1} * nn.W{i - 1}');
        end
        
        %calculate running exponential activations for use with sparsity
        if(nn.nonSparsityPenalty>0)

            nn.p{i} = 0.99 * nn.p{i} + 0.01 * mean(nn.a{i}, 1);
        end
    end
   
%update W
%     x = [ones(m,1) x];
%    H = nn.a{n-1}/(nn.a{n-1}'*nn.a{n-1});
    H = nn.a{n-1}*pinv(nn.a{n-1}'*nn.a{n-1});
    E = 2 * [ones(m,1) x]';
    G = (nn.a{n-1}.*(1-nn.a{n-1}')');
    J = (H*(nn.a{n-1}'*y)*(y'*H) - y*(y'*H));
    F = G.*J;
    T = E * F;
    nn.dW{1} = T';
    for i = 1 : (nn.n - 2)
        nn.dW{i} = alpha * nn.dW{i};
        if(momentum>0)
            nn.vW{i} = momentum*nn.vW{i} + nn.dW{i};
        end

        nn.W{i} = nn.W{i} - nn.vW{i};
    end   
    
 nn.a{1} = x;
 for i = 2 : n-1
        nn.a{i-1} = [ones(size(nn.a{i-1},1),1) nn.a{i-1}];
        switch nn.activation_function 
            case 'sigm'
                % Calculate the unit's outputs (including the bias term)
                nn.a{i} = sigm(nn.a{i - 1} * nn.W{i - 1}');
            case 'tanh_opt'
                nn.a{i} = tanh_opt(nn.a{i - 1} * nn.W{i - 1}');
        end
        
        %calculate running exponential activations for use with sparsity
        if(nn.nonSparsityPenalty>0)

            nn.p{i} = 0.99 * nn.p{i} + 0.01 * mean(nn.a{i}, 1);
        end
 end

%     %update U
%      nn.a{n-1} = [ones(size(nn.a{n-1},1),1) nn.a{n-1}];
%     A = nn.a{n-1}' * nn.a{n-1};
%     I =  eye(size(A,1));
% %     D = inv(A+opts.lamda*I);
%      D = pinv(A+opts.lamda*I);
% %    B = (A+opts.lamda*I) \ nn.a{n-1}';
%      B = D * nn.a{n-1}';
%     C = B * y;
%     nn.W{n - 1} = C;
%     nn.W{n - 1} = nn.W{n-1}'; 
    
     nn.a{n - 1} = [ones(size(nn.a{n-1},1),1) nn.a{n-1}];   
%     size(nn.a{n - 1})
%     size(nn.W{n - 1}')
    switch nn.output 
        case 'sigm'

            nn.a{n} = sigm(nn.a{n - 1} * nn.W{n - 1}');
        case 'linear'
            nn.a{n} = nn.a{n - 1} * nn.W{n - 1}';
        case 'softmax'
            nn.a{n} = nn.a{n - 1} * nn.W{n - 1}';
            nn.a{n} = exp(bsxfun(@minus, nn.a{n}, max(nn.a{n},[],2)));
            nn.a{n} = bsxfun(@rdivide, nn.a{n}, sum(nn.a{n}, 2)); 
    end

    %error and loss
    nn.e = y - nn.a{n};
    
    switch nn.output
        case {'sigm', 'linear'}
            nn.L = 1/2 * sum(sum(nn.e .^ 2)) / m; 
        case 'softmax'
            nn.L = -sum(sum(y .* log(nn.a{n}))) / m;
    end
end