%% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% 
% * Script 2: to classify EC and EO trials using the alpha power
% * Objective: to classify EEG trials of eyes open [EC] and eyes closed [EO]
% * Data Description: The example EEG data are from two sessions (EC and EO) 
%   of one subject. The number of channels is 64 and FCz is used as the 
%   refernece. Data are preprocessed with bandpass filtering (0.05-200Hz) 
%   and ICA denoising. Data are segmented into a seires of trials with a 
%   length of 200ms. The sampling rate is 250Hz, so the number of points of 
%   each trial is 50. There are 160 labeled trials (80 for EC and 80 for EO) 
%   and 80 unlabeled trials (actually, the first 40 are EC and others are EO). 
% 
%% Classification Model 1 %%
% to use alpha power as the feature
clear all; close all;
load eeg_features

%% STEP4. to select a subset of channels
chan_subset = [1:64]; % alpha at all channels are selected
% chan_subset = [9, 10, 64]; % alpha at Oz, O1, and O2 are selected
% chan_subset = 64; % alpha at Oz only is selected

Afeature_labeled = alpha_labeled(chan_subset,:).'; % only one feature
Afeature_unlabeled = alpha_unlabeled(chan_subset,:).'; % only one feature
Afeature_labeled_EC = Afeature_labeled(1:N_Labeled_EC,:);
Afeature_labeled_EO = Afeature_labeled(N_Labeled_EC+1:end,:);

%% STEP5. 10-fold cross-validation (CV) on labeled data using alpha power
K = 10; % K-fold CV
indices = crossvalind('Kfold',labeled_labels,K); % generate indices for CV
% NOTE: in this part, "train" and "test" mean the samples of labeled trials
for k = 1:K % K iterations
    cv_test_idx = find(indices == k); % indices for test samples in one round of CV
    cv_train_idx = find(indices ~= k); % indices for training samples in one round of CV
    cv_classout = classify(Afeature_labeled(cv_test_idx,:),Afeature_labeled(cv_train_idx,:),labeled_labels(cv_train_idx));
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

%% STEP6. classification on unlabeled data using alpha power
unlabeled_labels = [ones(N_Unlabeled/2,1);zeros(N_Unlabeled/2,1)]; % TRUE labels of unlabeled trials: 1 for EC; 0 for EO
classout = classify(Afeature_unlabeled,Afeature_labeled,labeled_labels,'linear');
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


