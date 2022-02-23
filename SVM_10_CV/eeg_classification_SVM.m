clear all; close all;

%% 1. load data and parameters
load eeg_classification.mat
% train_trials_ec: training trials (eyes-closed)
% train_trials_eo: training trials (eyes-open)
% test_trials_ec: test trials 
% fs = 250; % sampling rate
% channel_names = {'Oz'; 'O1'; 'O2'; 'PO'; 'PO3'; 'PO4'};
N_Chan = length(channel_names); % number of channels

%% 2. concatenate train trials and assign labels
train_trials = cat(3,train_trials_ec,train_trials_eo);
train_labels = [ones(size(train_trials_ec,3),1);zeros(size(train_trials_eo,3),1)]; % labels of training trials: 1 for ec; 0 for eo
N_Train = size(train_trials_ec,3) + size(train_trials_eo,3); % number of training trials
N_Test = size(test_trials,3); % number of test trials

%% 3. spectral estimation
nfft = 256; % Point of FFT
for n_chan=1:N_Chan
    for n_train=1:N_Train
        [P_train(:,n_chan,n_train),f] = pwelch(detrend(train_trials(:,n_chan,n_train)),[],[],nfft,fs);
    end % end of n_train
    for n_test=1:N_Test
        [P_test(:,n_chan,n_test),f] = pwelch(detrend(test_trials(:,n_chan,n_test)),[],[],nfft,fs);
    end % end of n_test
end % end of n_chan

% plot power spectra (the first sample of training data)
figure; image(P_train(:,:,1)', 'CDataMapping','scaled'); xlabel('Freq'); ylabel('Channel');

%% 4. feature extraction (alpha-band power in ec is significantly larger than in eo)
alpha_idx = find((f<=12)&(f>=8));  % frequency index of alpha band power
a_train = squeeze(mean(P_train(alpha_idx,:,:),1)); % extract alpha band power from train trials
a_test = squeeze(mean(P_test(alpha_idx,:,:),1)); % extract alpha band power from test trials
% outlier detection may be useful (check the distribution of features)

%% 5. dimension reduction using PCA

% SCORE is the PC matrix with each column being a PC (ranked by component variance)

% % PCA on whole-bands power spectra, along both frequencies and channels
% [COEFF_train,SCORE_train] = princomp((reshape(P_train,[size(P_train,1)*size(P_train,2),size(P_train,3)]))'); % PCA
% [COEFF_test,SCORE_test] = princomp((reshape(P_test,[size(P_test,1)*size(P_test,2),size(P_test,3)]))'); %PCA

% % Z scoring before PCA
% [Z_train,mu_train,sigma_train] = zscore((reshape(P_train,[size(P_train,1)*size(P_train,2),size(P_train,3)]))'); % standardize the input data to make it have zero mean and unit variance for each feature/variable
% [COEFF_train,SCORE_train] = princomp(Z_train);
% [Z_test,mu_test,sigma_test] = zscore((reshape(P_test,[size(P_test,1)*size(P_test,2),size(P_test,3)]))'); % standardize the input data to make it have zero mean and unit variance for each feature/variable
% [COEFF_test,SCORE_test] = princomp(Z_test);

% % PCA on alpha-band power along channels
% [COEFF_train,SCORE_train] = princomp(a_train.'); % PCA
% [COEFF_test,SCORE_test] = princomp(a_test.'); %PCA

% Z scoring before PCA
[Z_train,mu_train,sigma_train] = zscore(a_train.'); % standardize the input data to make it have zero mean and unit variance for each feature/variable
[COEFF_train,SCORE_train] = pca(Z_train);
[Z_test,mu_test,sigma_test] = zscore(a_test.'); % standardize the input data to make it have zero mean and unit variance for each feature/variable
[COEFF_test,SCORE_test] = pca(Z_test);

%% define features for classification

% % use the whole power spectra as features
% feature_train = (reshape(P_train,[size(P_train,1)*size(P_train,2),size(P_train,3)]))'; 
% feature_test = (reshape(P_test,[size(P_test,1)*size(P_test,2),size(P_test,3)]))'; 

% % use alpha-band power as features
feature_train = a_train'; 
feature_test = a_test'; 

% select the first D PCs as features
D = 2; % the reduced dimensionality
feature_train = SCORE_train(:,1:D);
feature_test = SCORE_test(:,1:D); 

%% 6. 10-fold cross-validation (CV) on training data
K = 10; % K-fold CV
indices = crossvalind('Kfold',train_labels,K); % generate indices for CV
% NOTE: in this part, "train" and "test" mean the samples in the training
% set which are used as training or test samples by CV
for k = 1:K % K iterations
    cv_test_idx = find(indices == k); % indices for test samples in one round of CV
    cv_train_idx = find(indices ~= k); % indices for training samples in one round of CV
    SVMStruct = fitcsvm(feature_train(cv_train_idx,:),train_labels(cv_train_idx));
    cv_classout = predict(SVMStruct,feature_train(cv_test_idx,:));
    cv_acc(k) = mean(cv_classout==train_labels(cv_test_idx)); % calculate accuracy
    TP = sum((cv_classout==train_labels(cv_test_idx))&(cv_classout==1)); % calculate True Positive
    TN = sum((cv_classout==train_labels(cv_test_idx))&(cv_classout==0)); % calculate True Negative
    FP = sum((cv_classout~=train_labels(cv_test_idx))&(cv_classout==1)); % calculate False Positive
    FN = sum((cv_classout~=train_labels(cv_test_idx))&(cv_classout==0)); % calculate False Negative
    cv_sensitivity(k) = TP/(TP+FN); % calculate specificity for detecting ec
    cv_specificity(k) = TN/(TN+FP); % calculate sensitivity for detecting ec
end
cv_acc_avg = mean(cv_acc); % averaged accuracy
cv_sensitivity_avg = mean(cv_sensitivity);  % averaged sensitivity for detecting ec
cv_specificity_avg = mean(cv_specificity);  % averaged specificity for detecting ec

%% 7. classification on test data
SVMStruct = fitcsvm(feature_train,train_labels);
classout = predict(SVMStruct,feature_test);
test_labels = [ones(100,1);zeros(100,1)]; % true labels of test trials: 1 for ec; 0 for eo
acc = mean(classout==test_labels); % calculate accuracy
TP = sum((classout==test_labels)&(classout==1)); % calculate True Positive
TN = sum((classout==test_labels)&(classout==0)); % calculate True Negative
FP = sum((classout~=test_labels)&(classout==1)); % calculate False Positive
FN = sum((classout~=test_labels)&(classout==0)); % calculate False Negative
sensitivity = TP/(TP+FN);% calculate sensitivity for detecting ec
specificity = TN/(TN+FP);% calculate specificity for detecting ec