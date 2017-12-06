function rbm = rbmsetup(architecture)

        rbm.sizes = architecture;
        rbm.numepochs =  50;
        rbm.batchsize = 5;
        rbm.momentum  =  0.6;
        rbm.alpha     =   0.1;
        rbm.penalty = 0.05;
    
        rbm.W  = 0.1*rand(rbm.sizes(2), rbm.sizes(1));
        rbm.vW = zeros(rbm.sizes(2), rbm.sizes(1));

        rbm.b  = zeros(rbm.sizes(1), 1);
        rbm.vb = zeros(rbm.sizes(1), 1);

        rbm.c  = zeros(rbm.sizes(2), 1);
        rbm.vc = zeros(rbm.sizes(2), 1);
end