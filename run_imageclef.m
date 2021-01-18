

%% Loading Data:
clearvars;
t0 = clock;
addpath('./utils/');
data_dir = './data/image_CLEF/';
domains = {'c','i','p'};
% ---lamda=0.01, gamma=1.0000, beta=0.0010, eta=0.0100, sigma=1.00, d1=256, d2=40 ---
options.lambda = 0.01;
options.gamma = 1.0000;
options.beta = 0.0010;
options.eta = 0.0100;
options.sigma = 1.00;
T = 11;
options.ReducedDim = 256;
d = 40;

count = 0;
for source_domain_index = 1:length(domains)
    load([data_dir 'imageclef-' domains{source_domain_index} '-resnet50-noft']);
    domainS_features_ori = L2Norm(resnet50_features);
    domainS_labels = labels+1;
    for target_domain_index = 1:length(domains)
        if target_domain_index == source_domain_index
            continue;
        end
        fprintf('Source domain: %s, Target domain: %s\n',domains{source_domain_index},domains{target_domain_index});
        load([data_dir 'imageclef-' domains{target_domain_index} '-resnet50-noft']);
        domainT_features = L2Norm(resnet50_features);
        domainT_labels = labels+1;
        
        %% use PCA
        X = double([domainS_features_ori;domainT_features]);
        P_pca = PCA(X,options);
        domainS_features = domainS_features_ori*P_pca;
        domainT_features = domainT_features*P_pca; 
        domainS_features = L2Norm(domainS_features);
        domainT_features = L2Norm(domainT_features);
        num_class = length(unique(domainT_labels));   
        
        %% Proposed method:
       [acc, acc_per_class] = CDEM(domainS_features,domainS_labels,domainT_features,domainT_labels,d,T,options);
          
       count = count + 1;
       all_acc_per_class(count,:) = mean(acc_per_class,2);
       all_acc_per_image(count,:) = acc;
    end
end
mean_acc_per_class = mean(all_acc_per_class,1);
mean_acc_per_image = mean(all_acc_per_image,1);
% ------输出每个taskT次迭代最高准确率和6个task平均准确率------
acc_per_task = max(all_acc_per_image,[],2);           
fprintf('\n---lamda=%0.4f, gamma=%0.4f, beta=%0.4f, eta=%0.4f, sigma=%0.2f, d1=%d, d2=%d ---\n',options.lambda, options.gamma,options.beta,options.eta,options.sigma, options.ReducedDim, d);
acc_per_task
mean_acc_all_task = mean(acc_per_task)   
TimeCost=etime(clock,t0);
fprintf('Time Cost %.2f seconds.', TimeCost);
% exit();
