
function [acc, acc_per_class] = CDEM(domainS_features,domainS_labels,domainT_features,domainT_labels,d,T,options)
num_iter = T;
options.ReducedDim = d;
options.alpha = 1;

num_class = length(unique(domainS_labels));
W_all = zeros(size(domainS_features,1)+size(domainT_features,1));
W_s = constructW1(domainS_labels);
W = W_all;
W(1:size(W_s,1),1:size(W_s,2)) =  W_s;
% looping
p = 1;
predLabels = [];
pseudoLabels = [];
for iter = 1:num_iter
	% 计算P矩阵的function
    P = constructP(domainS_features,domainS_labels,domainT_features,pseudoLabels, W,options);
    domainS_proj = domainS_features*P;
    domainT_proj = domainT_features*P;
    proj_mean = mean([domainS_proj;domainT_proj]);
    domainS_proj = domainS_proj - repmat(proj_mean,[size(domainS_proj,1) 1 ]);
    domainT_proj = domainT_proj - repmat(proj_mean,[size(domainT_proj,1) 1 ]);
    domainS_proj = L2Norm(domainS_proj);
    domainT_proj = L2Norm(domainT_proj);
    %% distance to class means
    classMeans = zeros(num_class,options.ReducedDim);
    for i = 1:num_class
        classMeans(i,:) = mean(domainS_proj(domainS_labels==i,:));
    end
    classMeans = L2Norm(classMeans);
    distClassMeans = EuDist2(domainT_proj,classMeans);
    targetClusterMeans = vgg_kmeans(double(domainT_proj'), num_class, classMeans')';
    targetClusterMeans = L2Norm(targetClusterMeans);
    distClusterMeans = EuDist2(domainT_proj,targetClusterMeans);
    expMatrix = exp(-distClassMeans);
    expMatrix2 = exp(-distClusterMeans);
    probMatrix1 = expMatrix./repmat(sum(expMatrix,2),[1 num_class]);
    probMatrix2 = expMatrix2./repmat(sum(expMatrix2,2),[1 num_class]);
    
    probMatrix = probMatrix1 * (1-iter./num_iter) + probMatrix2 * iter./num_iter;
    [prob,predLabels] = max(probMatrix');
    
    %% 挑选p1和p2预测class一样的类
    [~,I1] = max(probMatrix1');
    [~,I2] = max(probMatrix2');
    samePredict = find(I1 == I2); % P1 P2预测相等的下标集合
    prob1 = prob(samePredict);  % 取出这些预测一致样本的概率
    predLabels1 = predLabels(samePredict);  % 取出这些预测一致样本的预测标签
    
    p=iter/num_iter;
    p = max(p,0);
    [sortedProb,index] = sort(prob1);  % 对预测一致样本的预测概率排序，得到的index对应samePredict的下标
    sortedPredLabels = predLabels1(index);
    trustable = zeros(1,length(prob1));
    %% 从每个类中按照预设条件和类平衡思想挑选样本
    for i = 1:num_class
        ntc = length(find(predLabels==i));
        ntc_same = length(find(predLabels1 == i));
        % 要从预测一致样本中找当前class，注意二者index要一致，现在都是samePredict中的下标
        thisClassProb = sortedProb(sortedPredLabels==i);
        if length(thisClassProb)>0
            %从每个类中按照预设条件和类平衡思想挑选出min(iter/num_iter * nc, sameDc)个样本
            minProb = thisClassProb(max(ntc_same-(floor(p*ntc)+1) , 1));
            % 找出预测一致样本中预测值大于最小预测阈值的样本，注意，得到的是samePredict中的下标
            trustable = trustable+ (prob1>minProb).*(predLabels1==i);
        end
    end
    % 找到真正对应目标域样本的index
    true_index = samePredict(trustable==1);
    pseudoLabels = predLabels;
    trustable = zeros(1, length(prob));
    trustable(true_index) = 1;
    pseudoLabels(~trustable) = -1;
    
    W = constructW1([domainS_labels,pseudoLabels]);
	% ----------------------------------------
    %% calculate ACC
    acc(iter) = sum(predLabels==domainT_labels)/length(domainT_labels);
    for i = 1:num_class
        acc_per_class(iter,i) = sum((predLabels == domainT_labels).*(domainT_labels==i))/sum(domainT_labels==i);
    end
    fprintf('Iteration=%d/%d, Acc:%0.3f,Mean acc per class: %0.3f\n', iter,num_iter, acc(iter), mean(acc_per_class(iter,:)));
    if sum(trustable)>=length(prob)
        break;
    end
end
