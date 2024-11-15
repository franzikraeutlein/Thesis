# Compute SNR
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import os
import mne


data_path = '/Users/cntlab/Desktop/Thesis/Output/'
out_path = '/Users/cntlab/Desktop/Thesis/Output/SNR'

time = np.arange(0.5,1.501,0.001)


########################          Computation Functions                 ###################################

# Computes SNR for one subject in one session either for full trial (set tw_duration = 'full') or sepecific 
# time segments in seconds (e.g., tw_duration=0.25)
def compute_SNR(session, sub, tw_duration, main_channels_only=True):
    freqs = np.arange(13,120,2) # freqs for which Power was computed
    fbands = ['Beta', 'Gamma', 'Gamma1', 'Gamma2']
    sfreq = 1000
    if session == 'EEG':
        epochs_hand_EEG = mne.read_epochs(os.path.join(data_path, 'Data_clean', 'EEG', 'hand_clean_sess1_P'+str(sub)+'_epo.fif'), preload=True, verbose='Warning')
        ch_names = epochs_hand_EEG.ch_names
        power_hand = np.load(os.path.join(data_path, 'PowerEEG', 'Sub'+str(sub)+'power_hand_EEG.npy'), allow_pickle=True) # trials x chs x freqs x time
        power_rest = np.load(os.path.join(data_path, 'PowerEEG', 'Sub'+str(sub)+'power_rest_EEG.npy'), allow_pickle=True) # trials x chs x freqs x time
        selected_channels_names = np.array(['FC1', 'C3', 'CP1'])
        selected_channels_idx = np.nonzero([ch_name in selected_channels_names for ch_name in ch_names])[0]
        
        if sub == 3 or sub == 12: # for these two participants data mistakenly sampled with 500 instead of 1000Hz
            sfreq = 500

    if session == 'OPM':
        power_hand = np.load(os.path.join(data_path, 'Power_OPM', 'Sub'+str(sub)+'power_hand_OPM.npy'), allow_pickle=True) # trials x chs x freqs x time
        power_rest = np.load(os.path.join(data_path, 'Power_OPM', 'Sub'+str(sub)+'power_rest_OPM.npy'), allow_pickle=True) # trials x chs x freqs x time
        selected_channels_names = np.array([2,3,4])
        selected_channels_idx = selected_channels_names-1
    
    if main_channels_only:
        power_hand = power_hand[:,selected_channels_idx,:,:]
        power_rest = power_rest[:,selected_channels_idx,:,:]
        
    
    if tw_duration == 'full':
        tws = [1*sfreq]
        tw_duration = 2
    else:
        tstep= int(sfreq*tw_duration)
        tws = np.arange(0, power_hand.shape[3]-tstep,tstep)
        
    snr = np.empty((len(fbands), power_hand.shape[1],len(tws)))
    for count,fband in enumerate(fbands):
        if fband == 'Beta':
            indices_fband = np.where((freqs >= 13)&(freqs <= 30))[0]
        if fband == 'Gamma':
            indices_fband = np.where((freqs >= 30)&(freqs<=60))[0]
        if fband == 'Gamma1':
            indices_fband = np.where((freqs >= 60)&(freqs<=90))[0]
        if fband == 'Gamma2':
            indices_fband = np.where((freqs >= 90)&(freqs<=120))[0]

        # Power Data in respective fband
        power_hand_fband = power_hand[:,:,indices_fband,:]
        power_rest_fband = power_rest[:,:,indices_fband,:]
    
        # compute SNR over time-periods of length tw_duration (s)
        power_hand_tw = np.empty((power_hand_fband.shape[0],power_hand_fband.shape[1],power_hand_fband.shape[2], len(tws)))
        power_rest_tw = np.empty((power_rest_fband.shape[0],power_rest_fband.shape[1],power_rest_fband.shape[2], len(tws)))
        for count_tw,tw in enumerate(tws):
            if tw_duration == 2:
                hand_tw = power_hand_fband[:,:,:,tw:-1]
                rest_tw = power_rest_fband[:,:,:,tw:-1]
            else:
                hand_tw = power_hand_fband[:,:,:,tw:tw+tstep]
                rest_tw = power_rest_fband[:,:,:,tw:tw+tstep]
        
            # average across time
            mean_hand_tw = np.mean(hand_tw, axis=3)
            mean_rest_tw = np.mean(rest_tw, axis=3)
        
            power_hand_tw[:,:,:,count_tw] = mean_hand_tw
            power_rest_tw[:,:,:,count_tw] = mean_rest_tw

        # Average over freqs
        power_hand_meanTime = np.mean(power_hand_tw, axis=2) # shape trial x channels
        power_rest_meanTime = np.mean(power_rest_tw, axis=2)
    
        # Compute difference (Due to preprocessing, conditions do not have the same number of trials)
        trial_diff = power_hand_meanTime.shape[0]-power_rest_meanTime.shape[0]
        if trial_diff >= 0: # mehr hand trials als rest --> reduce trial number of hand
            diff = power_hand_meanTime[trial_diff:,:,:] - power_rest_meanTime
            #diff = (power_hand_meanTime[trial_diff:,:,:] - power_rest_meanTime)/power_rest_meanTime*100
        if trial_diff < 0:
            diff = power_hand_meanTime - power_rest_meanTime[:trial_diff,:,:]
            #diff = (power_hand_meanTime - power_rest_meanTime[:trial_diff,:,:])/power_rest_meanTime[:trial_diff,:,:]*100


        mean_diff = np.mean(diff, axis=0)
        std_diff = np.std(diff, axis=0)

        snr_fband = mean_diff/std_diff
        snr[count,:,:] = snr_fband
    return snr, tw_duration


# Computes averages of SNR either across the three selected channels or across the time segments
# To see exact SNR for the given participant set print_values = True
def compute_meanSNR_fband(sub, session,tw_duration, over_channel=False, over_time=True, print_values=False):
    snr, tw_duration_real = compute_SNR(session=session, sub=sub, tw_duration=tw_duration)
    mean_fband_ch = np.round(np.mean(snr, axis=2),3)
    mean_fband_tw = np.round(np.mean(snr, axis=1),3)
    overall_mean = np.round(np.mean(snr, axis=(1,2)),3)
    num_chs = snr.shape[1]
    num_tws = snr.shape[2]
    num_fband = snr.shape[0]
    time_windows = np.round(np.arange(-1,2,tw_duration_real),2)
    if num_fband == 4:
        fband_names = ['Beta Band (13-30Hz): ', 'Gamma Band (30-60Hz): ', 'High Gamma Band 1 (60-90Hz): ', 'High Gamma Band 2 (90-120Hz): ']
    else:
        fband_names = []
        for fband in range(num_fband):
            fband_names.append('Frequency Band '+str(fband+1)+': ')
    if over_channel and over_time:
        if print_values:
            print('Mean was computed over '+str(num_chs)+' Channels and '+str(num_tws)+' Time windows for '+str(num_fband)+' Frequency Bands.')
            for fband in range(num_fband):
                print(fband_names[fband]+str(overall_mean[fband]))
        return overall_mean[fband]
    if over_time:
        if print_values:
            print('Mean was computed over '+str(num_tws)+' Time windows for '+str(num_fband)+' Frequency Bands.')
            for fband in range(num_fband):
                print(fband_names[fband])
                [print('      Channel '+str(ch)+': '+str(mean_fband_ch[fband,ch])) for ch in range(num_chs)]
        return mean_fband_ch
    if over_channel:
        if print_values:
            print('Mean was computed over '+str(num_chs)+' Channels for '+str(num_fband)+' Frequency Bands.')
            for fband in range(num_fband):
                print(fband_names[fband])
                [print('      Time Window '+str(tw+1)+' ('+str(time_windows[tw])+'s - '+str(np.round(time_windows[tw]+tw_duration,2))+'s): '+str(mean_fband_tw[fband,tw])) for tw in range(num_tws)]
        return mean_fband_tw, time_windows


# Saves SNR computation for full trial segment for all participants
def save_snr_allSubs(session):
    subs = [1,2,3,4,6,8,10,11,12]
    snr_all = np.empty((4,1,len(subs)))
    for count,sub in enumerate(subs):
        print('Compute SNR Subject '+str(sub)+'...')
        snr_sub, tw_sub = compute_meanSNR_fband(sub=sub, session=session,tw_duration='full', over_channel=True, over_time=False)
        snr_all[:,:,count] = snr_sub

    np.save(os.path.join(data_path, 'SNR_noTW_allSubs_'+session+'.npy'), snr_all)

# Saves SNR computation for individual 250ms time segments for all participants
def save_snr_allSubs_tw(tw_duration=0.25):
    print('This can take a few minutes.')
    print('Compute SNR Subject 1...')
    snr_eeg, time_windows = compute_meanSNR_fband(sub=1, session='EEG', tw_duration=tw_duration, over_channel=True, over_time=False)
    snr_opm, time_windows = compute_meanSNR_fband(sub=1, session='OPM', tw_duration=tw_duration, over_channel=True, over_time=False)
    snr_eeg_allS = np.empty((snr_eeg.shape[0], snr_eeg.shape[1], 6))
    snr_opm_allS = np.empty((snr_opm.shape[0], snr_opm.shape[1], 6))
    snr_eeg_allS[:,:,0] = snr_eeg
    snr_opm_allS[:,:,0] = snr_opm
    for count, s in enumerate([2,3,4,6,8,10,11,12]):
        print('Compute SNR Subject '+str(s)+'...')
        snr_eeg_allS[:,:,count+1], time_window = compute_meanSNR_fband(sub=s, session='EEG', tw_duration=tw_duration, over_channel=True, over_time=False, print_values=False)
        snr_opm_allS[:,:,count+1], time_window = compute_meanSNR_fband(sub=s, session='OPM', tw_duration=tw_duration, over_channel=True, over_time=False, print_values=False)
    print('Save data...')
    np.save(os.path.join(data_path, 'SNR_250_allSubs_OPM_.npy'), snr_opm_allS)
    np.save(os.path.join(data_path, 'SNR_250_allSubs_EEG_.npy'), snr_eeg_allS)



###########################           Plotting Functions               #######################################


# Plots SNR of full trial as boxplot to show descriptive statistics of SNR across participants
def plot_snr_allSubs_noTW(fig, exclude_last3=True):
    snr_eeg_all = np.load(os.path.join(data_path, 'SNR_noTW_allSubs_EEG.npy'))
    snr_opm_all = np.load(os.path.join(data_path, 'SNR_noTW_allSubs_OPM.npy'))

    if exclude_last3:
        snr_eeg_all = snr_eeg_all[:,:,:-3].T
        snr_opm_all = snr_opm_all[:,:,:-3].T
    else:
        snr_eeg_all = snr_eeg_all.T
        snr_opm_all = snr_opm_all.T

    num_subs = snr_eeg_all.shape[0]
    snr_eeg_all1 = snr_eeg_all.reshape((num_subs,4))
    snr_opm_all1 = snr_opm_all.reshape((num_subs,4))
    
    ax_opm = fig.add_subplot(2,2,3)
    ax_eeg = fig.add_subplot(2,2,4)
    ax_opm.boxplot(snr_opm_all1[:,:], notch=False, vert=True)
    ax_eeg.boxplot(snr_eeg_all1[:,:], notch=False, vert=True)
    ax_opm.set_xticklabels(labels=['Beta Band \n(13-30Hz)', 'Gamma Band \n(30-60Hz)', 'High Gamma \n Band 1 \n(60-90Hz)', 'High Gamma \n Band 2 \n(90-120Hz)'])
    ax_eeg.set_xticklabels(labels=['Beta Band \n(13-30Hz)', 'Gamma Band \n(30-60Hz)', 'High Gamma \n Band 1 \n(60-90Hz)', 'High Gamma \n Band 2 \n(90-120Hz)'])
    ax_opm.set_ylabel('SNR')
    ax_eeg.set_ylabel('SNR')
    return fig


# Creates plot for the SNR across individual time segments in the four frequency bands
# Either for individual subjectes (e.g., sub=1) or as averages across participants (sub='all')
def plot_SNR_worker(sub, snr_opm, snr_eeg, time_windows):
    if len(snr_opm)==2:
        sem_opm = snr_opm[1]
        snr_opm = snr_opm[0]
        sem_eeg = snr_eeg[1]
        snr_eeg = snr_eeg[0]
    fband_marker = ['o', 'P', 'v', 's']
    fband_colors = ['mediumslateblue', 'darkorange', 'orangered', 'firebrick']
    fband_names = ['Beta Band (13-30Hz)', 'Gamma Band (30-60Hz)', 'High Gamma Band 1 (60-90Hz)', 'High Gamma Band 2 (90-120Hz)']
    min_eeg = np.min(snr_eeg)
    max_eeg = np.max(snr_eeg)
    min_opm = np.min(snr_opm)
    max_opm = np.max(snr_opm)
    fig = plt.figure(figsize=(20,15))
    legend_elements=[]
    if sub == 'all':
        ax_opm = fig.add_subplot(2,2,1)
        ax_eeg = fig.add_subplot(2,2,2)
    else:
        ax_opm = fig.add_subplot(1,2,1)
        ax_eeg = fig.add_subplot(1,2,2)
    diff_tws = np.diff(time_windows)[0]
    x_axis = time_windows + diff_tws/2
    for fband in range(4):
        ax_opm.plot(x_axis, snr_opm[fband,:], color=fband_colors[fband], marker=fband_marker[fband])
        ax_eeg.plot(x_axis, snr_eeg[fband,:], color=fband_colors[fband], marker=fband_marker[fband])
        legend_elements.append(Line2D([0], [2], color=fband_colors[fband], marker=fband_marker[fband],label=fband_names[fband], lw=1))
    ax_opm.vlines(time_windows, np.repeat(min_opm, len(time_windows)), np.repeat(max_opm, len(time_windows)), ls='dashed', color='black', lw=0.5, alpha=0.4)
    ax_opm.vlines(time_windows+diff_tws, np.repeat(min_opm, len(time_windows)), np.repeat(max_opm, len(time_windows)), ls='dashed', color='black', lw=0.5, alpha=0.4)
    ax_eeg.vlines(time_windows, np.repeat(min_eeg, len(time_windows)), np.repeat(max_eeg, len(time_windows)), ls='dashed', color='black', lw=0.5, alpha=0.4)
    ax_eeg.vlines(time_windows+diff_tws, np.repeat(min_eeg, len(time_windows)), np.repeat(max_eeg, len(time_windows)), ls='dashed', color='black', lw=0.5, alpha=0.4)
    ax_opm.hlines(0, time_windows[0], time_windows[-1]+diff_tws, color='black', lw=0.7)
    ax_eeg.hlines(0, time_windows[0], time_windows[-1]+diff_tws, color='black', lw=0.7)
    ax_opm.set_xlabel('Time Windows [s]')
    ax_eeg.set_xlabel('Time Windows [s]')
    ax_opm.set_ylabel('SNR')
    ax_eeg.set_ylabel('SNR')
    ax_opm.set_title('OPM')
    ax_eeg.set_title('EEG')
    ax_opm.legend(handles=legend_elements, loc=3, prop={'size': 6}, title ='Frequency Bands')
    ax_eeg.legend(handles=legend_elements, loc=3, prop={'size': 6}, title ='Frequency Bands')
    if sub == 'all':
        fig.suptitle('SNR: Mean across Participants')
    else:
        fig.suptitle('Sub '+str(sub)+': SNR')
    return fig


# Full Figure of 1 participant, computes SNR and plots SNR for individual time segments
def plot_SNR_1Sub(sub, tw_duration):
    snr_eeg, time_windows = compute_meanSNR_fband(sub=sub, session='EEG', tw_duration=tw_duration, over_channel=True, over_time=False, print_values=False)
    snr_opm, time_windows = compute_meanSNR_fband(sub=sub, session='OPM', tw_duration=tw_duration, over_channel=True, over_time=False, print_values=False)
    fig = plot_SNR_worker(sub=sub, snr_opm=snr_opm, snr_eeg=snr_eeg, time_windows=time_windows)
    fig.show()

# Full Figure, loads SNR of all subjects and plots average time segment SNR in upper panel
# And boxplot of full trial SNR underneath
# Corresponds to Figure 9 in thesis
# Exclude last three participants from averages by setting exclude_last3 = True
def plot_SNR_AllSubs(exclude_last3 = True):    
    snr_eeg_all = np.load(os.path.join(data_path, 'SNR_250_allSubs_EEG_.npy'))
    snr_opm_all = np.load(os.path.join(data_path, 'SNR_250_allSubs_OPM_.npy'))
    snr_eeg1, time_windows = compute_meanSNR_fband(sub=1, session='EEG', tw_duration=0.25, over_channel=True, over_time=False, print_values=True)
    # Compute mean across subs:
    if exclude_last3:
        snr_eeg_all = snr_eeg_all[:,:,:-3]
        snr_opm_all = snr_opm_all[:,:,:-3]
    snr_eeg_mean = np.mean(snr_eeg_all, axis=2)
    snr_opm_mean = np.mean(snr_opm_all, axis=2)

    fig = plot_SNR_worker(sub='all', snr_opm=snr_opm_mean, snr_eeg=snr_eeg_mean, time_windows=time_windows)
    fig = plot_snr_allSubs_noTW(fig, exclude_last3 = exclude_last3)
    fig.show()
    
    
    
    
    
    
########################################    Run Functions   ###################################################

# For Figure of all Participants:
save_snr_allSubs('OPM')
save_snr_allSubs('EEG')


# Plot
# Requires SNR data of all participants computed and saved with above functions
plot_SNR_AllSubs(exclude_last3=True)

# Plot of individual Subject
plot_SNR_1Sub(sub=1, tw_duration=0.25)

# Return SNR values
compute_meanSNR_fband(sub=8,session='OPM',tw_duration=0.25, over_channel=True, over_time=False,print_values=True)



