# Imports
import os
import mne
import numpy as np

# Compute and save power for all subjects
def compute_power(session):
    for sub in [1,2,3,4,6,8,10,11,12]:
        print('Subject '+str(sub))
        freqs = np.arange(13,120,2)
        if session== 'OPM':
            data_path = '/Users/cntlab/Desktop/Thesis/Output/Data_clean'
            out_path = '/Users/cntlab/Desktop/Thesis/Output'
            factor = 10**30
        if session == 'EEG':
            data_path = '/Users/cntlab/Desktop/Thesis/Output/Data_clean/EEG'
            out_path = '/Users/cntlab/Desktop/Thesis/Output'
            factor = 10**12
        epochs_hand = mne.read_epochs(os.path.join(data_path, 'hand_clean_sess1_P'+str(sub)+'_epo.fif'), preload=True, verbose='Warning')
        epochs_rest = mne.read_epochs(os.path.join(data_path, 'rest_clean_sess1_P'+str(sub)+'_epo.fif'), preload=True, verbose='Warning')
        print('Compute Power ...')
        power_hand = epochs_hand.compute_tfr(method='multitaper', freqs=freqs, tmin=-1.5, tmax=2.5, n_cycles=freqs*0.32, time_bandwidth=3.2)
        power_rest = epochs_rest.compute_tfr(method='multitaper', freqs=freqs, tmin=-1.5, tmax=2.5, n_cycles=freqs*0.32, time_bandwidth=3.2)
        print('get power Data ...')
        power_data = power_hand.get_data()*factor  # trials x channels x freqs x time
        power_rest_data = power_rest.get_data()*factor
        print ('start Baseline Correction ...')
        if session == 'EEG' and (sub == 3 or sub == 12): # sampling rate 500 instead of 1000 Hz
            power_crop = power_data[:,:,:,250:-250] # -1 bis 2
            power_rest_crop = power_rest_data[:,:,:,250:-250]
            baseline = power_crop[:,:,:,0:150] # 300ms baseline period (-1 bis -0.7)
            baseline_rest = power_rest_crop[:,:,:,0:150]
        else:
            power_crop = power_data[:,:,:,500:-500] # -1 bis 2
            power_rest_crop = power_rest_data[:,:,:,500:-500]
            baseline = power_crop[:,:,:,0:300] # 300ms baseline period (-1 bis -0.7)
            baseline_rest = power_rest_crop[:,:,:,0:300]
        
        del power_hand,power_data, epochs_hand
        del power_rest, power_rest_data, epochs_rest
    
        # average power per frequency in 300ms baseline period
        baseline = np.mean(baseline, axis=3) #per trial, channel and freq
        baseline_rest = np.mean(baseline_rest, axis=3) #per trial, channel and freq
    
        baseline_4d_mean = baseline[:,:,:,np.newaxis]
        baseline_4d_mean_rest = baseline_rest[:,:,:,np.newaxis]

        print('Compute Baseline corrected power ...')
        power_b = power_crop / baseline_4d_mean  # trial-wise baseline korrigiert
        power_b_rest = power_rest_crop / baseline_4d_mean_rest  # trial-wise baseline korrigiert
    
        del power_crop, baseline, baseline_4d_mean
        del power_rest_crop, baseline_rest, baseline_4d_mean_rest
  
        print('Save output ...')
        if session == 'EEG':
            out_path_suffix = 'PowerEEG'
        if session == 'OPM':
            out_path_suffix = 'Power_OPM'
        np.save(os.path.join(out_path, out_path_suffix, 'Sub'+str(sub)+'power_hand_'+session), power_b)
        np.save(os.path.join(out_path, out_path_suffix, 'Sub'+str(sub)+'power_rest_'+session), power_b_rest)
    return

compute_power('OPM')
# For each subject load Power Data for three main channels used for analysis
# Downsample to 500 Hz
# two Datasets: one, where means across trials for each subject are stacked together
# The other dataset contains all trials of all subjects
def get_Power_allSubs(session):
    data_path = '/Users/cntlab/Desktop/Thesis/Output/'
    #for sub in [1,2,3,4,6,8,10,11,12]:
    for sub in [1,2,3,4,6,8,10,11,12]:
        print(sub)
        if session == 'EEG':
            epochs_hand_EEG = mne.read_epochs(os.path.join(data_path, 'Data_clean', 'EEG', 'hand_clean_sess1_P'+str(sub)+'_epo.fif'), preload=True, verbose='Warning')
            ch_names = epochs_hand_EEG.ch_names
            selected_channels_eeg = np.array(['FC1', 'C3', 'CP1'])
            selected_channels = np.nonzero([ch_name in selected_channels_eeg for ch_name in ch_names])[0]
            power_hand = np.load(os.path.join(data_path, 'PowerEEG', 'Sub'+str(sub)+'power_hand_EEG.npy'), allow_pickle=True) # trials x chs x freqs x time
            power_rest = np.load(os.path.join(data_path, 'PowerEEG', 'Sub'+str(sub)+'power_rest_EEG.npy'), allow_pickle=True) # trials x chs x freqs x time
        if session == 'OPM':
            selected_channels = np.array([1,2,3])
            power_hand = np.load(os.path.join(data_path, 'Power_OPM', 'Sub'+str(sub)+'power_hand_OPM.npy'), allow_pickle=True) # trials x chs x freqs x time
            power_rest = np.load(os.path.join(data_path, 'Power_OPM', 'Sub'+str(sub)+'power_rest_OPM.npy'), allow_pickle=True) # trials x chs x freqs x time
        # Stack power across subjects
        if sub == 1:
            power_hand_all = power_hand[:,selected_channels,:,::2]
            power_rest_all = power_rest[:,selected_channels,:,::2]
        
            power_hand_mean_all = np.mean(power_hand[:,selected_channels,:,:], axis=0)[np.newaxis,:,:,::2]
            power_rest_mean_all = np.mean(power_rest[:,selected_channels,:,:], axis=0)[np.newaxis,:,:,::2]
        elif (sub==3 or sub == 12) and session=='EEG':
            power_hand_all = np.vstack([power_hand_all, power_hand[:,selected_channels,:,:]])
            power_rest_all = np.vstack([power_rest_all, power_rest[:,selected_channels,:,:]])
        
            power_hand_mean_all = np.vstack([power_hand_mean_all, np.mean(power_hand[:,selected_channels,:,:],axis=0)[np.newaxis,:,:,:]])
            power_rest_mean_all = np.vstack([power_rest_mean_all, np.mean(power_rest[:,selected_channels,:,:],axis=0)[np.newaxis,:,:,:]])
        else:
            power_hand_all = np.vstack([power_hand_all, power_hand[:,selected_channels,:,::2]])
            power_rest_all = np.vstack([power_rest_all, power_rest[:,selected_channels,:,::2]])
        
            power_hand_mean_all = np.vstack([power_hand_mean_all, np.mean(power_hand[:,selected_channels,:,:],axis=0)[np.newaxis,:,:,::2]])
            power_rest_mean_all = np.vstack([power_rest_mean_all, np.mean(power_rest[:,selected_channels,:,:],axis=0)[np.newaxis,:,:,::2]])
            
    if session == 'EEG':
        out_path_suffix = 'PowerEEG'
    if session == 'OPM':
        out_path_suffix = 'Power_OPM'
    # Save power averages of all Subjects
    np.save(os.path.join(data_path, out_path_suffix, 'AllSubs_meanT_power_hand_'+session), power_hand_mean_all[:,:,:,250:])
    np.save(os.path.join(data_path, out_path_suffix, 'AllSubs_meanT_power_rest_'+session), power_rest_mean_all[:,:,:,250:])
    # Save Power for all subjects and all their trials
    np.save(os.path.join(data_path, out_path_suffix, 'AllSubs_indT_power_hand_'+session), power_hand_all[:,:,:,250:])
    np.save(os.path.join(data_path, out_path_suffix, 'AllSubs_indT_power_rest_'+session), power_rest_all[:,:,:,250:])



compute_power(session='EEG')

get_Power_allSubs(session='EEG')

