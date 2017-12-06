function y = pca(mixedsig)

%����˵����y = pca(mixedsig)��������mixedsigΪ n*T �׻�����ݾ���nΪ�źŸ�����TΪ��������
% yΪ m*T ������������
% n��ά����T����������

if nargin == 0
    error('You must supply the mixed data as input argument.');
end
if length(size(mixedsig))>2
    error('Input data can not have more than two dimensions. ');
end
if any(any(isnan(mixedsig)))
    error('Input data contains NaN''s.');
end

%����������������������������ȥ��ֵ������������������������
meanValue = mean(mixedsig')';
[m,n] = size(mixedsig);
%mixedsig = mixedsig - meanValue*ones(1,size(meanValue)); %�����ݱ���ά���ܴ�ʱ���׳���Out of memory
for s =  1:m
    for t = 1:n
        mixedsig(s,t) = mixedsig(s,t) - meanValue(s);
    end
end
[Dim,NumofSampl] = size(mixedsig);
oldDimension = Dim;
fprintf('Number of signals: %d\n',Dim);
fprintf('Number of samples: %d\n',NumofSampl);
fprintf('Calculate PCA...');
firstEig = 1;
lastEig = Dim;
covarianceMatrix = corrcoef(mixedsig');    %����Э�������
[E,D] = eig(covarianceMatrix);          %����Э������������ֵ����������

%����������Э������������ֵ������ֵ�ĸ���lastEig������
%rankTolerance = 1;
%maxLastEig = sum(diag(D) >= rankTolerance);
%lastEig = maxLastEig;
lastEig = 100;

%��������������������������������ֵ��������������������
eigenvalues = flipud(sort(diag(D)));

%������������������ȥ����С������ֵ��������������������
if lastEig < oldDimension
    lowerLimitValue = (eigenvalues(lastEig) + eigenvalues(lastEig + 1))/2;
else
    lowerLimitValue = eigenvalues(oldDimension) - 1;
end
lowerColumns = diag(D) > lowerLimitValue;

%����������ȥ���ϴ������ֵ(һ��û����һ��)������������
if firstEig > 1
    higherLimitValue = (eigenvalues(firstEig - 1) + eigenvalues(firstEig))/2;
else
    higherLimitValue = eigenvalues(1) + 1;
end
higherColumns = diag(D) < higherLimitValue;

%�������������������ϲ�ѡ�������ֵ��������������������
selectedColumns =lowerColumns & higherColumns; %

%�������������������������Ľ����Ϣ������������������
fprintf('Selected [%d] dimensions.\n',sum(selectedColumns));
fprintf('Smallest remaining (non-zero) eigenvalue[ %g ]\n',eigenvalues(lastEig));
fprintf('Largest remaining (non-zero) eigenvalue[ %g ]\n',eigenvalues(firstEig));
fprintf('Sum of removed eigenvalue[ %g ]\n',sum(diag(D) .* (~selectedColumns)));

%��������������ѡ����Ӧ������ֵ������������������������
E = selcol(E,selectedColumns);
D = selcol(selcol(D,selectedColumns)',selectedColumns);

%������������������������׻����󡪡�������������������
whiteningMatrix = inv(sqrt(D)) * E';
dewhiteningMatrix = E * sqrt(D);

%����������������������ȡ������������������������������
y = whiteningMatrix * mixedsig;

%����������������������ѡ���ӳ��򡪡�������������������
function newMatrix = selcol(oldMatrix,maskVector)
if size(maskVector,1)~= size(oldMatrix,2)
    error('The mask vector and matrix are of uncompatible size.');
end
numTaken = 0;
for i = 1:size(maskVector,1)
    if maskVector(i,1) == 1  
        takingMask(1,numTaken + 1) = i;
        numTaken = numTaken + 1;
    end
end
newMatrix = oldMatrix(:,takingMask);