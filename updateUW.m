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
   
%% update W   batch-mode gradient descent
epsilon = 1.0e-6;
% Store the number of decision variables
x0 = [-1; 0.2; 0.1; 2; 1; 2; 3];
nDecVar = length(x0);
nIter = length(x);
idxSG = nn.a{n-1};
% Allocate output
xMat = zeros(nDecVar, nIter + 1);

% Set the initial guess
xMat(:, 1) = x0;

% Repeat `idxSG` if it has fewer columns than `nIter`
if size(idxSG, 2) < nIter
    idxSG = repmat(idxSG, 1, ceil(nIter/size(idxSG, 2)));
    idxSG(:, nIter + 1 : 1 : end) = [];
end

% Initialise accumulator variables
accG = zeros(nDecVar, 1); % accumulated gradients
accD = zeros(nDecVar, 1); % accumulated updates (deltas)

% Run optimisation
for i = 1 : 1 : nIter
    % Get gradients w.r.t. stochastic objective at the current iteration
    sgCurr = StochGrad(idxSG(:, i), xMat(:, i));
    
    % Update accumulated gradients
    accG = 0.95.*accG + (1 - 0.95).*(sgCurr.^2);
    
    % Compute update
    dCurr = -(sqrt(accD + epsilon)./sqrt(accG + epsilon)).*sgCurr;
    
    % Update accumulated updates (deltas)
    accD = 0.95.*accD + (1 - 0.95).*(dCurr.^2);
    
    % Update decision variables
    xMat(:, i + 1) = xMat(:, i) + dCurr;
    nn.dW{i} = xMat(:, i + 1);
end
    
 nn.a{1} = x;
 for i = 2 : n-1
        nn.a{i-1} = [ones(size(nn.a{i-1},1),1) nn.a{i-1}];
        switch nn.activation_function 
            case 'sigm'
                %Calculate the unit's outputs (including the bias term)
                nn.a{i} = sigm(nn.a{i - 1} * nn.W{i - 1}');
            case 'tanh_opt'
                nn.a{i} = tanh_opt(nn.a{i - 1} * nn.W{i - 1}');
        end
        
        % calculate running exponential activations for use with sparsity
        if(nn.nonSparsityPenalty>0)

            nn.p{i} = 0.99 * nn.p{i} + 0.01 * mean(nn.a{i}, 1);
        end
 end
 

%%     %update U
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