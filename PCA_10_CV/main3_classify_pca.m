%% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% 
% * Script 3: to classify EC and EO trials using PCA-transformed features
% * Objective: to classify EEG trials of eyes open [EC] and eyes closed [EO]
% * Data Description: The example EEG data are from two sessions (EC and EO) 
%   of one subject. The number of channels is 64 and FCz is used as the 
%   refernece. Data are preprocessed with bandpass filtering (0.05-200Hz) 
%   and ICA denoising. Data are segmented into a seires of trials with a 
%   length of 200ms. The sampling rate is 250Hz, so the number of points of 
%   each trial is 50. There are 160 labeled trials (80 for EC and 80 for EO) 
%   and 80 unlabeled trials (actually, the first 40 are EC and others are EO). 
% 
%% Classification Model 2 %%
% to use PCA-transformed features
clear all; close all;
load eeg_features

%% STEP7. dimension reduction using PCA along channels
D = 1; % the new dimension (D < N_Chan). Try some values like 1, 4, 7, 10
[Z_labeled,mu_labeled,sigma_labeled] = zscore(alpha_labeled.'); % standardize the input data to make it have zero mean and unit variance for each feature/variable
[COEFF_labeled,SCORE_labeled] = pca(alpha_labeled.'); % use PCA on labeled data
[Z_unlabeled,mu_unlabeled,sigma_unlabeled] = zscore(alpha_unlabeled.'); % standardize the input data to make it have zero mean and unit variance for each feature/variable
[COEFF_unlabeled,SCORE_unlabeled] = pca(alpha_unlabeled.'); % use PCA on unlabeled data
% SCORE is the PC matrix with each column being a PC (ranked by component variance)
PCfeature_labeled = SCORE_labeled(:,1:D); % only one feature
PCfeature_unlabeled = SCORE_unlabeled(:,1:D); % only one feature
PCfeature_labeled_EC = PCfeature_labeled(1:N_Labeled_EC,:);
PCfeature_labeled_EO = PCfeature_labeled(N_Labeled_EC+1:end,:);

%% FIG7. plot PCA features (1st component)
figure('units','normalized','position',[0    0.0556    1.0000    0.8361])
subplot(311);
stem(PCfeature_labeled_EC(:,1),'b','fill');
title('Labeled Trials [EC]','fontweight','bold')
xlabel('Trial/Sample'); ylabel('PC1')
set(gca,'YLim',[-100 100])
subplot(312);
stem(PCfeature_labeled_EO(:,1),'r','fill');
title('Labeled Trials [EO]','fontweight','bold')
xlabel('Trial/Sample'); ylabel('PC1')
set(gca,'YLim',[-100 100])
subplot(313);
stem(PCfeature_unlabeled(:,1),'k','fill');
title('Unlabeled Trials','fontweight','bold')
xlabel('Trial/Sample'); ylabel('PC1')
set(gca,'YLim',[-100 100])

%% STEP8. 10-fold cross-validation (CV) on labeled data using PC features
K = 10; % K-fold CV
indices = crossvalind('Kfold',labeled_labels,K); % generate indices for CV
% NOTE: in this part, "train" and "test" mean the samples of labeled trials
for k = 1:K % K iterations
    cv_test_idx = find(indices == k); % indices for test samples in one round of CV
    cv_train_idx = find(indices ~= k); % indices for training samples in one round of CV
    cv_classout = classify(PCfeature_labeled(cv_test_idx,:),PCfeature_labeled(cv_train_idx,:),labeled_labels(cv_train_idx));
    cv_acc(k) = mean(cv_classout==labeled_labels(cv_test_idx)); % calculate accuracy
    TP = sum((cv_classout==labeled_labels(cv_test_idx))&(cv_classout==1)); % calculate True Positive
    TN = sum((cv_classout==labeled_labels(cv_test_idx))&(cv_classout==0)); % calculate True Negative
    FP = sum((cv_classout~=labeled_labels(cv_test_idx))&(cv_classout==1)); % calculate False Positive
    FN = sum((cv_classout~=labeled_labels(cv_test_idx))&(cv_classout==0)); % calculate False Negative
    cv_sen(k) = TP/(TP+FN); % calculate sensitivity for detecting ec
    cv_spe(k) = TN/(TN+FP); % calculate specificity for detecting ec
end
labeled_acc = mean(cv_acc); % averaged accuracy
labeled_sen = mean(cv_sen);  % averaged sensitivity for detecting ec
labeled_spe = mean(cv_spe);  % averaged specificity for detecting ec
disp(['================================'])
disp(['% Results (CV) on Labeled Data %'])
disp(['  Accuracy:    ',num2str(labeled_acc*100,'%4.2f'),'%'])
disp(['  Sensitivity: ',num2str(labeled_sen*100,'%4.2f'),'%'])
disp(['  Specificity: ',num2str(labeled_spe*100,'%4.2f'),'%'])
disp(['================================'])

%% STEP9. classification on unlabeled data using PC features
unlabeled_labels = [ones(N_Unlabeled/2,1);zeros(N_Unlabeled/2,1)]; % TRUE labels of unlabeled trials: 1 for EC; 0 for EO
classout = classify(PCfeature_unlabeled,PCfeature_labeled,labeled_labels,'linear');
unlabeled_acc = mean(classout==unlabeled_labels); % calculate accuracy
TP = sum((classout==unlabeled_labels)&(classout==1)); % calculate True Positive
TN = sum((classout==unlabeled_labels)&(classout==0)); % calculate True Negative
FP = sum((classout~=unlabeled_labels)&(classout==1)); % calculate False Positive
FN = sum((classout~=unlabeled_labels)&(classout==0)); % calculate False Negative
unlabeled_sen = TP/(TP+FN); % calculate specificity for detecting ec
unlabeled_spe = TN/(TN+FP); % calculate sensitivity for detecting ec
disp(['================================'])
disp(['% Results on Unlabeled Data %'])
disp(['  Accuracy:    ',num2str(unlabeled_acc*100,'%4.2f'),'%'])
disp(['  Sensitivity: ',num2str(unlabeled_sen*100,'%4.2f'),'%'])
disp(['  Specificity: ',num2str(unlabeled_spe*100,'%4.2f'),'%'])
disp(['================================'])

%% FIG9. show classification results on unlabeled data using PC features
figure('units','normalized','position',[0    0.0556    1.0000    0.8361])
subplot(211);
stem(PCfeature_unlabeled(:,1),'k','fill');
title('Features of Unlabled Trials','fontweight','bold')
xlabel('Trial/Sample'); ylabel('PC1')
set(gca,'YLim',[-100 100])
subplot(212);
hold on; box on;
stem((classout-0.5)*2,'Marker','x','color','k')
stem([1:40],unlabeled_labels(1:40),'Marker','s','color','b')
stem([41:80],unlabeled_labels(41:80)-1,'Marker','s','color','r')
set(gca,'ytick',[-1 1],'yticklabel',{'EO','EC'},'ylim',[-1.5 1.5]);
title('Classification Results on Unlabeled Trials','fontweight','bold')
xlabel('Trial/Sample'); ylabel('Classes')
