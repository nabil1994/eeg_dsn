function [ dsn ] = dsnsetup( dsn, x, opts )


    n = size(x, 2);
    dsn.sizes = [n, dsn.sizes];
    if (~isfield(opts,'vis_units') || isempty(opts.vis_units))
        opts.vis_units = 'sigm';
    end;
    if (~isfield(opts,'hid_units') || isempty(opts.hid_units))
        opts.hid_units = 'sigm';
    end;
    if (~isfield(opts,'momentum') || isempty(opts.momentum))
        opts.momentum = 0;
    end;
    if (~isfield(opts,'cdn') || isempty(opts.momentum))
        opts.cdn = 1;
    end;

    for u = 1 : numel(dsn.sizes) - 1
        dsn.rbm{u}.alpha    = opts.alpha;
        dsn.rbm{u}.momentum = opts.momentum;
        dsn.rbm{u}.cdn      = opts.cdn;
        dsn.rbm{u}.penalty = opts.penalty;
        % make vis_units only actually visible units (1st layer)
        if (u == 1)
            dsn.rbm{u}.vis_units = opts.vis_units;
        else
            dsn.rbm{u}.vis_units = opts.hid_units;
        end;
        dsn.rbm{u}.hid_units = opts.hid_units;

%         dsn.rbm{u}.W  = zeros(dsn.sizes(u + 1), dsn.sizes(u));
         dsn.rbm{u}.W  = 0.1*rand(dsn.sizes(u + 1), dsn.sizes(u));
%         dsn.rbm{u}.W  = normrnd(0,0.01,dsn.sizes(u + 1), dsn.sizes(u));
        
        dsn.rbm{u}.vW = zeros(dsn.sizes(u + 1), dsn.sizes(u));
            
         dsn.rbm{u}.b  = zeros(dsn.sizes(u), 1);
%         dsn.rbm{u}.b  = 0.1*rand(dsn.sizes(u), 1);
%         dsn.rbm{u}.b = normrnd(0,0.01,dsn.sizes(u),1);
        dsn.rbm{u}.vb = zeros(dsn.sizes(u), 1);

         dsn.rbm{u}.c  = zeros(dsn.sizes(u + 1), 1);
%         dsn.rbm{u}.c  = 0.1*rand(dsn.sizes(u+1), 1);
%         dsn.rbm{u}.c = normrnd(0,0.01,dsn.sizes(u+1),1);
        dsn.rbm{u}.vc = zeros(dsn.sizes(u + 1), 1);
    end


end

