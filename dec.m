%滤波函数
function l_data=dec(s)
[C,L]=wavedec(s,4,'db5');%waverec函数是多尺度一维离散小波重构函数
ca4=appcoef(C,L,'db5',4);
%cd5=detcoef(C,L,4);
cd4=detcoef(C,L,4);
cd3=detcoef(C,L,3);
cd2=detcoef(C,L,2);
cd1=detcoef(C,L,1);
C1=[0*ca4',cd4',cd3',0*cd2',0*cd1'];
l_data=waverec(C1,L,'db5');