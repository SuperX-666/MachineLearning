%% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% 
% * Script 1: to extract features from raw EEG data
% * Objective: to classify EEG trials of eyes open [EC] and eyes closed [EO]
% * Data Description: The example EEG data are from two sessions (EC and EO) 
%   of one subject. The number of channels is 64 and FCz is used as the 
%   refernece. Data are preprocessed with bandpass filtering (0.05-200Hz) 
%   and ICA denoising. Data are segmented into a seires of trials with a 
%   length of 200ms. The sampling rate is 250Hz, so the number of points of 
%   each trial is 50. There are 160 labeled trials (80 for EC and 80 for EO) 
%   and 80 unlabeled trials (actually, the first 40 are EC and others are EO). 
% 
% by Zhiguo Zhang, School of Biomedical Engineering, Shenzhen University
% Email: zgzhang@szu.edu.cn

%% STEP1. load data and define parameters
clear all; close all;
load eeg_rawdata.mat
% labeled_trials_ec: labeled trials (eyes-closed, EC), dimension: #channel x #time points x #trials
% labeled_trials_eo: labeled trials (eyes-open, EO), dimension: #channel x #time points x #trials
% unlabeled_trials_ec: unlabeled trials, dimension: #channel x #time points x #trials 
% fs = 250: sampling rate
% channel_names: names of 64 channels (FCz is the reference)

N_Chan = length(channel_names); % number of channels = 64
N_Labeled_EC = size(labeled_trials_ec,3); % number of labeled EC trials = 80
N_Labeled_EO = size(labeled_trials_eo,3); % number of labeled EO trials = 80
N_Labeled = N_Labeled_EC + N_Labeled_EO; % number of ALL labeled trials = 160
N_Unlabeled = size(unlabeled_trials,3); % number of unlabeled trials = 80
N_Time = size(unlabeled_trials,2); % number of time points = 50
t = [1:N_Time]'/fs; % time axis

% concatenate labeled trials and assign labels
labeled_trials = cat(3,labeled_trials_ec,labeled_trials_eo); % ALL labeled trials
labeled_labels = [ones(N_Labeled_EC,1);zeros(N_Labeled_EC,1)]; % labels of labeled trials: 1 for EC; 0 for EO

%% FIG1. show waveforms of some trials at Oz
figure('units','normalized','position',[0    0.0556    1.0000    0.8361])
for n=1:5
    subplot(3,5,n)
    plot(t,labeled_trials_ec(64,:,n),'b')
    title(['Labeled Trial [EC] ',num2str(n)],'fontweight','bold')
    xlabel('Time (s)','verticalalignment','middle'); ylabel('Amplitude (\muV)','verticalalignment','middle'); axis tight
    set(gca,'xtick',[0.1 0.2 0.3 0.4 0.5])
    subplot(3,5,n+5)
    plot(t,labeled_trials_eo(64,:,n),'r')
    title(['Labeled Trial [EO] ',num2str(n)],'fontweight','bold')
    xlabel('Time (s)','verticalalignment','middle'); ylabel('Amplitude (\muV)','verticalalignment','middle'); axis tight
    set(gca,'xtick',[0.1 0.2 0.3 0.4 0.5])
    subplot(3,5,n+10)
    n_unlabeled = randi(N_Unlabeled,1);
    plot(t,unlabeled_trials(64,:,n_unlabeled),'k')
    title(['Unlabeled Trial ',num2str(n_unlabeled)],'fontweight','bold')
    xlabel('Time (s)','verticalalignment','middle'); ylabel('Amplitude (\muV)','verticalalignment','middle'); axis tight
    set(gca,'xtick',[0.1 0.2 0.3 0.4 0.5])
end

%% STEP2. spectral estimation
nfft = 256; % Point of FFT
for n_chan=1:N_Chan
    disp(['Spectral Estimation: channel ',channel_names{n_chan}, ' (',num2str(n_chan),'/64)'])
    for n_labeled=1:N_Labeled
        [P_labeled(n_chan,:,n_labeled),f] = periodogram(detrend(labeled_trials(n_chan,:,n_labeled)),[],nfft,fs);
    end % end of n_labeled
    for n_unlabeled=1:N_Unlabeled
        [P_unlabeled(n_chan,:,n_unlabeled),f] = periodogram(detrend(unlabeled_trials(n_chan,:,n_unlabeled)),[],nfft,fs);
    end % end of n_unlabeled
end % end of n_chan
P_labeled = 10*log10(P_labeled); % use the log scale for power
P_unlabeled = 10*log10(P_unlabeled); % use the log scale for power
P_Labeled_EC = P_labeled(:,:,1:N_Labeled_EC);
P_Labeled_EO = P_labeled(:,:,N_Labeled_EC+1:end);

%% FIG2a. show spectra of some trials at Oz
f_idx = find((f>0)&(f<=50));
figure('units','normalized','position',[0    0.0556    1.0000    0.8361])
for n=1:5
    ax(n)=subplot(3,5,n);
    plot(f(f_idx),P_Labeled_EC(64,f_idx,n),'b')
    title(['Labeled Trial [EC] ',num2str(n)],'fontweight','bold')
    xlabel('Frequency (Hz)','verticalalignment','middle'); ylabel('Power (dB)','verticalalignment','middle'); axis tight
    set(gca,'xtick',[4 8 12 30 50],'xgrid','on')
    ax(n+5)=subplot(3,5,n+5);
    plot(f(f_idx),P_Labeled_EO(64,f_idx,n),'r')
    title(['Labeled Trial [EO] ',num2str(n)],'fontweight','bold')
    xlabel('Frequency (Hz)','verticalalignment','middle'); ylabel('Power (dB)','verticalalignment','middle'); axis tight
    set(gca,'xtick',[4 8 12 30 50],'xgrid','on')
    ax(n+10)=subplot(3,5,n+10);
    n_unlabeled = randi(N_Unlabeled,1);
    plot(f(f_idx),P_unlabeled(64,f_idx,n_unlabeled),'k')
    title(['Unlabeled Trial ',num2str(n_unlabeled)],'fontweight','bold')
    xlabel('Frequency (Hz)','verticalalignment','middle'); ylabel('Power (dB)','verticalalignment','middle'); axis tight
    set(gca,'xtick',[4 8 12 30 50],'xgrid','on')
end
linkaxes(ax,'xy')

%% FIG2b. show averaged spectra of EC vs EO at all channels  
f_idx = find((f>0)&(f<=50));
chan_idx = [1,61,2,11,3,17,4,12,13,5,18,6,14,15,7,19,8,16,9,64,10];
subplot_idx = [2:4,6:20,22:24];
figure('units','normalized','position',[0    0.0556    1.0000    0.8361])
for n_chan=1:length(chan_idx)
        ax(n_chan)=subplot(5,5,subplot_idx(n_chan));
        hold on; box on;
        plot(f(f_idx),mean(P_Labeled_EC(chan_idx(n_chan),f_idx,:),3),'b')
        plot(f(f_idx),mean(P_Labeled_EO(chan_idx(n_chan),f_idx,:),3),'r')
        title(channel_names{chan_idx(n_chan)},'fontweight','bold')
        xlabel('Frequency (Hz)','verticalalignment','middle'); ylabel('Power (dB)','verticalalignment','middle'); axis tight
        set(gca,'xtick',[4 8 12 30 50])
        legend({'EC','EO'});legend('boxoff')
end
%linkaxes(ax,'xy')

%% STEP3. feature extraction (alpha power in EC is larger than that in EO)
alpha_idx = find((f<=12)&(f>=8));  % frequency index of alpha band power
alpha_labeled = squeeze(mean(P_labeled(:,alpha_idx,:),2)); % extract alpha band power from labeled trials
alpha_unlabeled = squeeze(mean(P_unlabeled(:,alpha_idx,:),2)); % extract alpha band power from unlabeled trials
alpha_Labeled_EC = alpha_labeled(:,1:N_Labeled_EC);
alpha_Labeled_EO = alpha_labeled(:,N_Labeled_EC+1:end);
% outlier detection may be useful (check the distribution of features)

save eeg_features % save data and features

%% FIG3. plot features (alpha power at Oz)
figure('units','normalized','position',[0    0.0556    1.0000    0.8361])
subplot(311);
stem(alpha_Labeled_EC(64,:),'b','fill');
title('Labeled Trials [EC]','fontweight','bold')
xlabel('Trial/Sample'); ylabel('Alpha Power at Oz')
set(gca,'YLim',[-10 30])
subplot(312);
stem(alpha_Labeled_EO(64,:),'r','fill');
title('Labeled Trials [EO]','fontweight','bold')
xlabel('Trial/Sample'); ylabel('Alpha Power at Oz')
set(gca,'YLim',[-10 30])
subplot(313);
stem(alpha_unlabeled(64,:),'k','fill');
title('Unlabeled Trials','fontweight','bold')
xlabel('Trial/Sample'); ylabel('Alpha Power at Oz')
set(gca,'YLim',[-10 30])

