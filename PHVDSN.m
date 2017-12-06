clear all;
close all;
clc;

load subject4;
data = [train_data;test_data];

train_num = size(train_label,1);%训练样本数
test_num = size(test_label,1);%测试样本数
sample_num = train_num + test_num;%总的样本数
sample = 1000;%一组样本点数

Fs=250; %自己设置采样频率
Num=1000;  %自己设置采样点数
NFFT = 2^nextpow2(Num);%转化为2的基数倍
f=Fs/2*linspace(0,1,NFFT/2); %求出FFT转化频率

fp1=8;fp2=30; %通带
fs1=5;fs2=35; %阻带
rp=1;rs=30;
wp=[2*fp1/Fs,2*fp2/Fs];
ws=[2*fs1/Fs,2*fs2/Fs];
[N,wso]=cheb2ord(wp,ws,rp,rs);
[b,a]=cheby2(N,rs,wso);
for i = 1:sample_num
x_r=(i-1)*sample+126:i*sample-625; %取提示开始后0.5-1.5s的数据
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
%% 求频谱
% Y_C3=fft(data_C3,NFFT)/Num; %进行FFT变换
% Y_Cz=fft(data_Cz,NFFT)/Num;
% Y_C4=fft(data_C4,NFFT)/Num;

% Y_C3=fft(data(x_r,1),NFFT)/Num; %进行FFT变换
% Y_Cz=fft(data(x_r,1),NFFT)/Num;
% Y_C4=fft(data(x_r,1),NFFT)/Num;
% Data(i,:) = [Y_C3(:,31:140) Y_Cz(:,31:140) Y_C4(:,31:140)]; %   
% figure(2);
% % plot(f,2*abs(Y_C3(1:NFFT/2)),'r',f,2*abs(Y_C4(1:NFFT/2)),'b');title('脑电信号频域图');xlabel('Frequency');ylabel('频谱值');
% plot(1:NFFT,Y_C3);
end

% Data = pca1(Data');
% Data = Data';
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
opts.numepochs =   50;

opts.batchsize = 10;
opts.momentum  =   0.6;
opts.alpha     =   0.1;
opts.penalty = 0.05;
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
nn.numepochs =  3;
nn.batchsize = 10;
%train nn
[nn] = nntrainpso(nn, train_x, train_y, opts);
%nn = phvdsnupdate(nn, train_x, train_y, opts); 
[er, bad, a, b] = hvdsntest(nn, test_x, test_y);
min_er = er;
phvdsn = nn;
for i = 2 : 5
%     dsn = dsntrain(dsn, train_x, opts);
    nn = piledhvdsns( nn, dsn );
    opts.stack     =   i;
    [nn] = nntrainpso(nn, train_x, train_y, opts);
%   nn = phvdsnupdate(nn, train_x, train_y, opts); 
    [er, bad, a, b] = hvdsntest(nn, test_x, test_y);
    if er < min_er 
        min_er = er;
        phvdsn = nn;
    end    
end
disp(['Test error: ',num2str(min_er)]);
