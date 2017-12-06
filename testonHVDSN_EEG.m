clear all;
close all;
clc;

load subject8;
data = [train_data;test_data];

train_num = size(train_label,1);%训练样本数
test_num = size(test_label,1);%测试样本数
sample_num = train_num + test_num;%总样本数
sample = 1000;%一组样本点数

Fs=250; %自己设置采样频率
fp1=8;fp2=30; %通带
fs1=5;fs2=35; %阻带
rp=1;rs=30;
wp=[2*fp1/Fs,2*fp2/Fs];
ws=[2*fs1/Fs,2*fs2/Fs];
[N,wso]=cheb2ord(wp,ws,rp,rs);
[b,a]=cheby2(N,rs,wso);
for i = 1:sample_num;
x_r=(i-1)*sample+126:i*sample-375; %取提示开始后0.5-2.5s的数据
%% 去均值
data(x_r,1)=remmean(data(x_r,1));
data(x_r,2)=remmean(data(x_r,2));
data(x_r,3)=remmean(data(x_r,3));

%% 小波滤波
data_C3=dec(data(x_r,1)); 
data_Cz=dec(data(x_r,2));
data_C4=dec(data(x_r,3));

% data_C3=filter(b,a,data(x_r,1))';
% data_Cz=filter(b,a,data(x_r,2))';
% data_C4=filter(b,a,data(x_r,3))';
Data(i,:) = [data_C3 data_Cz data_C4]; % 
end
% train_y = train_label;
% test_y = test_label;
% train_x = Data(1:train_num,:);
% test_x = Data(train_num+1:end,:);
train_y = [train_label;test_label(1:test_num-70,:)];
test_y = test_label(test_num-70+1:end,:);
train_x = Data(1:sample_num-70,:);
test_x = Data(sample_num-70+1:end,:);
%% 归一化
train_x = abs(mapminmax(train_x, 0, 1));
test_x = abs(mapminmax(test_x, 0, 1));

ClassNum = 2;
m = size(train_y, 1);
n = size(test_y,1);
train_y = full(sparse(1:m, train_y, 1, m, ClassNum));
test_y = full(sparse(1:n, test_y, 1, n, ClassNum));
%%  ex2 train a 100-100 hidden unit dsn and use its weights to initialize a NN
rand('state',0)
%train dsn
dsn.sizes = [20];
opts.numepochs =   10;
opts.batchsize = 10;
opts.momentum  =   0.6;
opts.alpha     =   0.1;
opts.penalty = 0.01;
opts.lamda = 0.5;
opts.stack     =   0;
opts.cdn = 1;
opts.vis_units  = 'sigm';   % type of visible units (default: 'sigm')
opts.hid_units  = 'sigm';   % type of hidden units  (default: 'sigm')
                            % units can be 'sigm' - sigmoid, 'linear' - linear
                            % 'NReLU' - noisy rectified linear (Gaussian noise)
dsn = dsnsetup(dsn, train_x, opts);  
dsn = dsntrain(dsn, train_x, opts);  %训练各层受限玻尔兹曼机

%unfold dsn to nn
nn = dsnunfoldtonn(dsn, 2);  %将训练好的RBM权重用于初始化神经网络权重
nn.activation_function = 'sigm';
nn.output = 'sigm';
%train nn
nn.numepochs =  3;
nn.batchsize = 10;
% opts.momentum = 0.5;
% opts.alpha = 0.1;
nn = dsnupdate(nn, train_x, train_y, opts); %更新神经网络权重
[er] = hvdsntest(nn, test_x, test_y)
min_er = er;
hvdsn = nn;
for i = 2 : 5
    dsn = dsntrain(dsn, train_x, opts);
    nn = piledhvdsns( nn, dsn );
    opts.stack     =   i;
    nn = dsnupdate(nn, train_x, train_y, opts);
    [er] = hvdsntest(nn, test_x, test_y)
    if er < min_er 
        min_er = er;
        hvdsn = nn;
    end    
end

