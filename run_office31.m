
clearvars;
t0 = clock;
addpath('./utils/');
% data_dir = '/root/cyh/final-domain-adaptation-capls-master/data/Office31/';
data_dir = './data/Office31/';
domains = {'A','D','W'};
options.beta = 0.01;
options.lambda = 0.1;
options.gamma = 0.001;
options.eta = 0.0001;
options.sigma = 0.1;
fprintf('\n\n---beta=%0.4f, lamda=%0.4f, gamma=%0.4f, eta=%0.4f, sigma=%0.4f ---\n',options.beta,options.lambda, options.gamma,options.eta,options.sigma);

count = 0;
for source_domain_index = 1:length(domains)
    load([data_dir 'office-' domains{source_domain_index} '-resnet50-noft']);
    domainS_features_ori = L2Norm(resnet50_features);
    domainS_labels = labels+1;
    
    for target_domain_index = 1:length(domains)
        if target_domain_index == source_domain_index
            continue;
        end
        fprintf('Source domain: %s, Target domain: %s\n',domains{source_domain_index},domains{target_domain_index});
        load([data_dir 'office-' domains{target_domain_index} '-resnet50-noft']);
        domainT_features = L2Norm(resnet50_features);
        domainT_labels = labels+1;
        
        opts.ReducedDim = 256;
        X = double([domainS_features_ori;domainT_features]);
        P_pca = PCA(X,opts);
        domainS_features = domainS_features_ori*P_pca;
        domainT_features = domainT_features*P_pca;
        domainS_features = L2Norm(domainS_features);
        domainT_features = L2Norm(domainT_features);
        num_class = length(unique(domainT_labels));
       
        %% Proposed method:
        d = 128;
        T = 11;
        [ acc,acc_per_class ]= CDE(domainS_features,domainS_labels,domainT_features,domainT_labels,d,T,options);%change by cyh
        count = count + 1;
        all_acc_per_class(count,:) = mean(acc_per_class,2);
        all_acc_per_image(count,:) = acc;
    end
end
acc_per_task = max(all_acc_per_image,[],2);
acc_per_task
mean_acc_all_task = mean(acc_per_task)
TimeCost=etime(clock,t0);
fprintf('Time Cost %.2f seconds.', TimeCost);
