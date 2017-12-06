function [nn, L]  = phvdsnupdate(nn, train_x, train_y, opts)
%NNTRAIN trains a neural net
% [nn, L] = nnff(nn, x, y, opts) trains the neural network nn with input x and
% output y for opts.numepochs epochs, with minibatches of size
% opts.batchsize. Returns a neural network nn with updated activations,
% errors, weights and biases, (nn.a, nn.e, nn.W, nn.b) and L, the sum
% squared error for each training minibatch.

assert(isfloat(train_x), 'train_x must be a float');
assert(nargin == 4,'number ofinput arguments must be 4 or 6')

m = size(train_x, 1);

batchsize = nn.batchsize;
numepochs = nn.numepochs;

numbatches = floor(m / batchsize);

%assert(rem(numbatches, 1) == 0, 'numbatches must be a integer');

L = zeros(numepochs*numbatches,1);
n = 1;
t1 = 0;
for i = 1 : numepochs
    tic;
    
    kk = randperm(m);
    for l = 1 : numbatches
        %batch_x = train_x(kk((l - 1) * batchsize + 1 : l * batchsize), :);
        batch_x = extractminibatch(kk,l,batchsize,train_x);
        
        %Add noise to input (for use in denoising autoencoder)
        if(nn.inputZeroMaskedFraction ~= 0)
            batch_x = batch_x.*(rand(size(batch_x))>nn.inputZeroMaskedFraction);
        end
        
        %batch_y = train_y(kk((l - 1) * batchsize + 1 : l * batchsize), :);
        batch_y = extractminibatch(kk,l,batchsize,train_y);
        
        nn = updateUW(nn, batch_x, batch_y,opts);
        
        if l ~= 1
            nn.label = [nn.label; nn.a{3}];
        else
            nn.label = nn.a{3};
        end
        
        L(n) = nn.L;
        
        n = n + 1;
    end
    
    t = toc;
    t1 = t1 + t;
    disp(['epoch ' num2str(i) '/' num2str(nn.numepochs) '.Took ' num2str(t)...
        ' seconds' '. Mean squared error on training set is ' num2str(mean(L((n-numbatches):(n-1))))]);
    nn.learningRate = nn.learningRate * nn.scaling_learningRate;    
    nn = nntrainpso(nn, train_x, train_y, opts);
end
    
t1
end

