function [nn]= nntrainpso(nn,x,y,opts)
%% �����ʼ��
%����Ⱥ�㷨�е���������
c1 = 2;
c2 = 2;

maxgen = 50;    %�����  
sizepop = 10;   %��Ⱥ��ģ
Vmax=0.5;
Vmin=-0.5;
popmax=1;
popmin=-1;
ws=0.9;
we=0.4;
[m,n] = size(nn.W{1});
% [a,b] = size(nn.W{2});
%%*******��Ȩֵ����չ����һ�е�����**********
W = [];
for i = 1:m
    W = [W nn.W{1}(i,:)];
end
% for i = 1:a
%     W = [W nn.W{2}(i,:)];
% end

    %% �����ʼ���Ӻ��ٶ�
    
    for i=1:sizepop
        %������һ����Ⱥ
        pop(i,:) = W;    %��ʼ��Ⱥ
        V(i,:)=0.5*rands(1,m*n);  %��ʼ���ٶ� + a*b
        %������Ӧ��
        nn = nnffpso(nn,x,y,pop(i,:));
        fitness(i) = nn.L;   %Ⱦɫ�����Ӧ��
    end

    %% ���弫ֵ��Ⱥ�弫ֵ
    [bestfitness bestindex]=min(fitness);
    zbest=pop(bestindex,:);   %ȫ�����
    gbest=pop;    %�������
    fitnessgbest=fitness;   %���������Ӧ��ֵ
    fitnesszbest=bestfitness;   %ȫ�������Ӧ��ֵ
    
    %% ���Ѱ��
    tic;
%       w = 0.5;
    for i=1:maxgen
    
       w=we*(ws/we)^(1+1/maxgen);
%          w = w - (ws-we)/maxgen;
        for j=1:sizepop
        
            %�ٶȸ���
            V(j,:) = w*V(j,:) + c1*rand*(gbest(j,:) - pop(j,:)) + c2*rand*(zbest - pop(j,:));
            V(j,find(V(j,:)>Vmax))=Vmax;
            V(j,find(V(j,:)<Vmin))=Vmin;

            %��Ⱥ����
            pop(j,:)=pop(j,:)+V(j,:);
            pop(j,find(pop(j,:)>popmax))=popmax;
            pop(j,find(pop(j,:)<popmin))=popmin;
        
            %��Ӧ��ֵ
            
           nn = nnffpso(nn,x,y,pop(j,:));
           fitness(j) = nn.L;   %Ⱦɫ�����Ӧ��

        end

        for j=1:sizepop

            %�������Ÿ���
            if fitness(j) < fitnessgbest(j)
                gbest(j,:) = pop(j,:);
                fitnessgbest(j) = fitness(j);
            end

            %Ⱥ�����Ÿ���
            if fitness(j) < fitnesszbest
                zbest = pop(j,:);
                fitnesszbest = fitness(j);
            end
        end 
         bestfitness(i)=fitnesszbest;   
    end
    t = toc
% figure();
% plot(bestfitness);xlabel('Iterations');ylabel('Fitness');    
% nn = nnffpso(nn,x,y,zbest);
for i = 1:m 
    nn.W{1}(i,:) = zbest(:,(i-1)*n+1:i*n);
end
% for i = 1:a 
%     nn.W{2}(i,:) = zbest(:,(i-1)*b + m*n + 1:i*b + m*n);
% end
q = nn.n;
nn.a{1} = x;
for j = 2 : q-1
        nn.a{j-1} = [ones(size(nn.a{j-1},1),1) nn.a{j-1}];
        switch nn.activation_function 
            case 'sigm'
                % Calculate the unit's outputs (including the bias term)
                nn.a{j} = sigm(nn.a{j - 1} * nn.W{j - 1}');
            case 'tanh_opt'
                nn.a{j} = tanh_opt(nn.a{j - 1} * nn.W{j - 1}');
        end
        
        %calculate running exponential activations for use with sparsity
        if(nn.nonSparsityPenalty>0)

            nn.p{j} = 0.99 * nn.p{j} + 0.01 * mean(nn.a{j}, 1);
        end
end
%update U
    nn.a{q-1} = [ones(size(nn.a{q-1},1),1) nn.a{q-1}];
    A = nn.a{q-1}' * nn.a{q-1};
    I =  eye(size(A,1));
%     D = inv(A+opts.lamda*I);
     D = pinv(A+opts.lamda*I);
%    B = (A+opts.lamda*I) \ nn.a{n-1}';
     B = D * nn.a{q-1}';
    C = B * y;
    nn.W{q - 1} = C;
    nn.W{q - 1} = nn.W{q-1}'; 
end

