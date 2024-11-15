# Create Figure 2
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import os
import mne
from matplotlib.lines import Line2D
from scipy import stats

data_path = '/Users/cntlab/Desktop/Thesis/Output/'
data_path_epochs = '/Users/cntlab/Desktop/Thesis/Output/Data_clean'
out_path = '/Users/cntlab/Desktop/Thesis/Output/'


##########################     Functions to get data to be plotted      ##############################################

### Individual Subjects

# Compute power spectral density for a given partcipant and session
def compute_psd(session, sub, selected_chs='all'):
    fmin = 13
    fmax = 120
    if session == 'OPM':
        epochs_hand = mne.read_epochs(os.path.join(data_path_epochs, 'hand_clean_sess1_P'+str(sub)+'_epo.fif'), preload=True, verbose='Warning')
        epochs_rest = mne.read_epochs(os.path.join(data_path_epochs, 'rest_clean_sess1_P'+str(sub)+'_epo.fif'), preload=True, verbose='Warning')
        factor = 10**30
        if selected_chs != 'all':
            selected_channels_idx = np.array(selected_chs)-1
    
    if session == 'EEG':
        epochs_hand = mne.read_epochs(os.path.join(data_path_epochs,'EEG', 'hand_clean_sess1_P'+str(sub)+'_epo.fif'), preload=True, verbose='Warning')
        epochs_rest = mne.read_epochs(os.path.join(data_path_epochs, 'EEG', 'rest_clean_sess1_P'+str(sub)+'_epo.fif'), preload=True, verbose='Warning')
        factor = 10**12
        ch_names = epochs_hand.ch_names
        if selected_chs != 'all':
            selected_channels_idx = np.nonzero([ch_name in selected_chs for ch_name in ch_names])[0]
    
    spectrum_hand_t = epochs_hand.compute_psd(method='multitaper', fmin=fmin, fmax=fmax, tmin=0.5, tmax=1.9, n_jobs=None)#, n_fft=1201)
    spectrum_hand_b = epochs_hand.compute_psd(method='multitaper', fmin=fmin, fmax=fmax, tmin=-1.5, tmax=-0.1, n_jobs=None)#, n_fft=1201)
    
    spectrum_rest_t = epochs_rest.compute_psd(method='multitaper', fmin=fmin, fmax=fmax, tmin=0.5, tmax=1.9, n_jobs=None)#, n_fft=1201)
    spectrum_rest_b = epochs_rest.compute_psd(method='multitaper', fmin=fmin, fmax=fmax, tmin=-1.5, tmax=-0.1, n_jobs=None)#, n_fft=1201)

    psds_hand_t, freqs_hand_t = spectrum_hand_t.get_data(return_freqs=True)
    psds_hand_t = (psds_hand_t*factor).astype(float)
    psds_rest_t = (spectrum_rest_t.get_data()*factor).astype(float)
    psds_hand_b = (spectrum_hand_b.get_data()*factor).astype(float)
    psds_rest_b = (spectrum_rest_b.get_data()*factor).astype(float)
    
    if selected_chs != 'all':
        ratio_hand = psds_hand_t[:,selected_channels_idx,:]/psds_hand_b[:,selected_channels_idx,:]
        ratio_rest = psds_rest_t[:,selected_channels_idx,:]/psds_rest_b[:,selected_channels_idx,:]
    else:
        ratio_hand = psds_hand_t/psds_hand_b
        ratio_rest = psds_rest_t/psds_rest_b
    
    #Decibel
    db_hand = 10*np.log10(ratio_hand)
    db_rest = 10*np.log10(ratio_rest)
    db_hand_avg = np.mean(db_hand, axis=0)
    db_rest_avg = np.mean(db_rest, axis=0)

    return [db_hand_avg, db_rest_avg],freqs_hand_t


# Compute event-related synchronization for specific participant and subject for all frequencies (13-120)
# Time segment can be specified with tmin and tmax
def compute_ers(sub,session, selected_chs='all', tmin=-1, tmax=2):
    if session == 'EEG':
        epochs_hand_EEG = mne.read_epochs(os.path.join(data_path, 'Data_clean', 'EEG', 'hand_clean_sess1_P'+str(sub)+'_epo.fif'), preload=True, verbose='Warning')
        power_hand = np.load(os.path.join(data_path, 'PowerEEG', 'Sub'+str(sub)+'power_hand_EEG.npy'), allow_pickle=True) # trials x chs x freqs x time
        power_rest = np.load(os.path.join(data_path, 'PowerEEG', 'Sub'+str(sub)+'power_rest_EEG.npy'), allow_pickle=True) # trials x chs x freqs x time
        ch_names = epochs_hand_EEG.ch_names
        if selected_chs != 'all':
            selected_channels_idx = np.nonzero([ch_name in selected_chs for ch_name in ch_names])[0]
        if sub == 3 or sub==12:
            sfreq = 500
        else:
            sfreq = 1000
        
    if session == 'OPM':
        power_hand = np.load(os.path.join(data_path, 'Power_OPM', 'Sub'+str(sub)+'power_hand_OPM.npy'), allow_pickle=True) # trials x chs x freqs x time
        power_rest = np.load(os.path.join(data_path, 'Power_OPM', 'Sub'+str(sub)+'power_rest_OPM.npy'), allow_pickle=True) # trials x chs x freqs x time
        sfreq = 1000
        if selected_chs != 'all':
            selected_channels_idx = np.array(selected_chs)-1
    
    ts = 1/sfreq
    if tmin < -1 or tmax > 2:
        print('Temporal limits out of bounds, must be between -1 and 2s. Full window length is assumed.')
        tmin = -1
        tmax = 2
    min_ds = int(tmin*sfreq - (-1)*sfreq)
    max_ds = int(tmax*sfreq - (-1)*sfreq)
    
    power_hand = power_hand[:,selected_channels_idx,:,min_ds:max_ds]
    power_rest = power_rest[:,selected_channels_idx,:,min_ds:max_ds]
    
    mean_hand = np.mean(power_hand, axis=0)
    mean_rest = np.mean(power_rest, axis=0)
    ers = (mean_hand-mean_rest)/mean_rest*100
    
    return ers


### Concatenated data of all Subjects


# Requires stacked power data of all Subs obtained via get_Power script
# Computes average ERS of all subjects for a given session
# include_last3 results from fact that last three participants demonstrated large movement artifacts in OPM data
def get_ers_allSubs(session, return_meanSem=True, include_last3=False, mean_over_chs=True):
    if session == 'EEG':
        power_hand = np.load(os.path.join(data_path, 'PowerEEG', 'AllSubs_meanT_power_hand_EEG.npy'), allow_pickle=True) # trials x chs x freqs x time
        power_rest = np.load(os.path.join(data_path, 'PowerEEG', 'AllSubs_meanT_power_rest_EEG.npy'), allow_pickle=True) # trials x chs x freqs x time
    if session == 'OPM':
        power_hand = np.load(os.path.join(data_path, 'Power_OPM', 'AllSubs_meanT_power_hand_OPM.npy'), allow_pickle=True) # trials x chs x freqs x time
        power_rest = np.load(os.path.join(data_path, 'Power_OPM', 'AllSubs_meanT_power_rest_OPM.npy'), allow_pickle=True) # trials x chs x freqs x time
    freqs = np.arange(13,120,2)
    indices_beta = np.where((freqs >= 13)&(freqs <= 30))[0]
    indices_gamma = np.where((freqs >= 60)&(freqs <= 90))[0]
    ers = (power_hand-power_rest)/power_rest*100
    ers_beta = ers[:,:,indices_beta,:]
    ers_gamma = ers[:,:,indices_gamma,:]
    
    if include_last3 == False:
        ers_beta = ers_beta[0:-3,:,:,:]
        ers_gamma = ers_gamma[0:-3,:,:,:]
    
    if mean_over_chs:
        ers_mean_ch_beta = np.mean(ers_beta, axis=(1,2))
        ers_mean_ch_gamma = np.mean(ers_gamma, axis=(1,2))
    else:
        ers_mean_ch_beta = np.mean(ers_beta, axis=2)
        ers_mean_ch_gamma = np.mean(ers_gamma, axis=2)
    
    mean_ers_beta = np.mean(ers_mean_ch_beta.astype(float), axis=0)
    mean_ers_gamma = np.mean(ers_mean_ch_gamma.astype(float), axis=0)
    
    sem_ers_beta = stats.sem(ers_mean_ch_beta.astype(float), axis=0)
    sem_ers_gamma = stats.sem(ers_mean_ch_gamma.astype(float), axis=0)
    if return_meanSem:
        return [mean_ers_beta, sem_ers_beta], [mean_ers_gamma, sem_ers_gamma]
    else:
        return ers_mean_ch_beta, ers_mean_ch_gamma


# Computes PSD values of all subjects for a given session, either averaged across trials (computeMean = True) or not
def psd_all_subs(session, computeMean=False):
    for sub in [1,2,3,4,6,8]:
        if session == 'OPM':
            selected_chs = [2,3,4]
        if session == 'EEG':
            selected_chs = ['FC1', 'C3', 'CP1']
        [db_hand_avg, db_rest_avg],freqs_hand_t= compute_psd(sub=sub, session=session,selected_chs=selected_chs)
        
        if sub == 1:
            psd_hand_all = db_hand_avg[np.newaxis,:,:]
            psd_rest_all = db_rest_avg[np.newaxis,:,:]
        else:
            psd_hand_all = np.vstack([psd_hand_all, db_hand_avg[np.newaxis,:,:]])
            psd_rest_all = np.vstack([psd_rest_all, db_rest_avg[np.newaxis,:,:]])
    
    if computeMean:
        psd_hand_all_mean = np.mean(psd_hand_all, axis=0)
        psd_rest_all_mean = np.mean(psd_rest_all, axis=0)
        return [psd_hand_all_mean, psd_rest_all_mean], freqs_hand_t
    else:
        return [psd_hand_all, psd_rest_all], freqs_hand_t


###############################         Plotting Functions           #############################

# Computes differences in power spectral densities between both conditions
# Differences are plotted for each selected channel of the given session in one plot
def plot_psd_diff(psd_data, freqs, session,fig):
    legend_elements_sensor=[]
    psd_hand = psd_data[0]
    psd_rest = psd_data[1]
    colors = ['slategrey', 'darkcyan', 'steelblue']
    title = session
    if session == 'OPM':
        ax = fig.add_subplot(3,2,5)
        ch_names = ['Sensor 2','Sensor 3','Sensor 4']
    if session == 'EEG':
        ax = fig.add_subplot(3,2,6)
        ch_names = ['FC1', 'C3', 'CP1']
    diff = psd_hand - psd_rest
    for sensor in range(3):
        legend_elements_sensor.append(Line2D([0], [2], color=colors[sensor], linestyle= 'solid',label=ch_names[sensor], lw=2))
        ax.plot(freqs, diff[sensor,:], color=colors[sensor], lw=1.5, ls='solid')
        ax.set_xlabel('Frequency [Hz]')
        ax.set_ylabel(title+' Recording \n dB')
    plt.hlines(0,freqs[0],freqs[-1],color='black',lw=0.7)
    #ax.legend(handles=legend_elements_sensor, loc=8, prop={'size': 6}, title ='Difference Movement-Rest \n               Sensors')
    ax.legend(handles=legend_elements_sensor, loc=4, prop={'size': 8}, title ='Sensors')

    return fig


# Plots Mean ERS of three slected channels for beta band (upper panel) and high gamma band (lower panel)
# for OPM data (left panels) and EEG data (right panels)
def plot_ers_allSubs():
    [ers_beta_EEG_mean, ers_beta_EEG_sem], [ers_gamma_EEG_mean, ers_gamma_EEG_sem] = get_ers_allSubs('EEG', return_meanSem=True, include_last3=True, mean_over_chs=False)
    [ers_beta_OPM_mean, ers_beta_OPM_sem], [ers_gamma_OPM_mean, ers_gamma_OPM_sem] = get_ers_allSubs('OPM', return_meanSem=True, include_last3=True, mean_over_chs=False)
    beta_all_EEG, gamma_all_EEG = get_ers_allSubs('EEG', return_meanSem=False, include_last3=True, mean_over_chs=False)
    beta_all_OPM, gamma_all_OPM = get_ers_allSubs('OPM', return_meanSem=False, include_last3=True, mean_over_chs=False)
    time_plot = np.arange(-0.5,2.002,0.002)
    colors = ['slategrey', 'darkcyan', 'steelblue']
    fig = plt.figure(figsize=(12,10))
    ax_beta_opm = fig.add_subplot(3,2,1)
    ax_beta_eeg = fig.add_subplot(3,2,2)
    ax_beta_eeg.sharey(ax_beta_opm)
    ax_gamma_opm = fig.add_subplot(3,2,3)
    ax_gamma_eeg = fig.add_subplot(3,2,4)
    ax_gamma_eeg.sharey(ax_gamma_opm)
    for sensor in range(3):
        ax_beta_eeg.plot(time_plot, ers_beta_EEG_mean[sensor], color=colors[sensor], lw=2, alpha=1)
        ax_beta_opm.plot(time_plot, ers_beta_OPM_mean[sensor], color=colors[sensor], lw=2, alpha=1)
        ax_gamma_eeg.plot(time_plot, ers_gamma_EEG_mean[sensor], color=colors[sensor], lw=2, alpha=1)
        ax_gamma_opm.plot(time_plot, ers_gamma_OPM_mean[sensor], color=colors[sensor], lw=2, alpha=1)
        for sub in range(6):
            ax_beta_eeg.plot(time_plot, beta_all_EEG[sub,sensor,:], color=colors[sensor], lw=0.8, ls=(0, (1, 1)), alpha=0.5)
            ax_beta_opm.plot(time_plot, beta_all_OPM[sub,sensor,:], color=colors[sensor], lw=0.8, ls=(0, (1, 1)), alpha=0.5)
            ax_gamma_eeg.plot(time_plot, gamma_all_EEG[sub,sensor,:], color=colors[sensor], lw=0.8, ls=(0, (1, 1)), alpha=0.5)
            ax_gamma_opm.plot(time_plot, gamma_all_OPM[sub,sensor,:], color=colors[sensor], lw=0.8, ls=(0, (1, 1)), alpha=0.5)
    plt.show()


# Overall Figure depicting mean beta and high gamma ERS and psd differences averaged across the three selected channels
def plot_ers_allSubs1ch():
    [ers_beta_EEG_mean, ers_beta_EEG_sem], [ers_gamma_EEG_mean, ers_gamma_EEG_sem] = get_ers_allSubs('EEG', return_meanSem=True, include_last3=True, mean_over_chs=False)
    [ers_beta_OPM_mean, ers_beta_OPM_sem], [ers_gamma_OPM_mean, ers_gamma_OPM_sem] = get_ers_allSubs('OPM', return_meanSem=True, include_last3=True, mean_over_chs=False)
    beta_all_EEG, gamma_all_EEG = get_ers_allSubs('EEG', return_meanSem=False, include_last3=True, mean_over_chs=True)
    beta_all_OPM, gamma_all_OPM = get_ers_allSubs('OPM', return_meanSem=False, include_last3=True, mean_over_chs=True)
    psd_data_EEG, freqs_EEG = psd_all_subs('EEG', True)
    psd_data_OPM, freqs_OPM = psd_all_subs('OPM', True)
    time_plot = np.arange(-0.5,2.002,0.002)
    colors = ['slategrey', 'darkcyan', 'steelblue']
    fig = plt.figure(figsize=(12,10))
    fig = plot_psd_diff(psd_data=psd_data_OPM, freqs=freqs_OPM,session='OPM', fig=fig)
    fig = plot_psd_diff(psd_data=psd_data_EEG, freqs=freqs_EEG, session='EEG', fig=fig)
    ax_beta_opm = fig.add_subplot(3,2,1)
    ax_beta_eeg = fig.add_subplot(3,2,2)
    ax_beta_eeg.sharey(ax_beta_opm)
    ax_gamma_opm = fig.add_subplot(3,2,3)
    ax_gamma_eeg = fig.add_subplot(3,2,4)
    ax_gamma_eeg.sharey(ax_gamma_opm)
    ax_beta_opm.set_title('OPM')
    ax_beta_eeg.set_title('EEG')
    ax_beta_opm.set_ylabel('OPM Recording \n Beta Band \nERS [%]')
    ax_beta_eeg.set_ylabel('EEG Recording \n Beta Band \nERS [%]')
    ax_beta_opm.set_xlabel('Time [s]')
    ax_beta_eeg.set_xlabel('Time [s]')
    ax_gamma_opm.set_xlabel('Time [s]')
    ax_gamma_eeg.set_xlabel('Time [s]')
    ax_gamma_opm.set_ylabel('OPM Recording \n High Gamma Band \nERS [%]')
    ax_gamma_eeg.set_ylabel('EEG Recording \n High Gamma Band \nERS [%]')
    for sensor in range(3):
        ax_beta_eeg.plot(time_plot, ers_beta_EEG_mean[sensor,:], color=colors[sensor], lw=2, alpha=1)
        ax_beta_opm.plot(time_plot, ers_beta_OPM_mean[sensor,:], color=colors[sensor], lw=2, alpha=1)
        ax_gamma_eeg.plot(time_plot, ers_gamma_EEG_mean[sensor,:], color=colors[sensor], lw=1.8, alpha=1)
        ax_gamma_opm.plot(time_plot, ers_gamma_OPM_mean[sensor,:], color=colors[sensor], lw=1.3, alpha=1)
    ax_beta_eeg.hlines(0, time_plot[0], time_plot[-1], color='black', lw=1)
    ax_beta_opm.hlines(0, time_plot[0], time_plot[-1], color='black', lw=1)
    ax_gamma_eeg.hlines(0, time_plot[0], time_plot[-1], color='black', lw=1)
    ax_gamma_opm.hlines(0, time_plot[0], time_plot[-1], color='black', lw=1)
    for sub in range(6):
        ax_beta_eeg.plot(time_plot, beta_all_EEG[sub,:], color='blue', lw=0.8, ls=(0, (1, 1)), alpha=0.5)
        ax_beta_opm.plot(time_plot, beta_all_OPM[sub,:], color='blue', lw=0.8, ls=(0, (1, 1)), alpha=0.5)
        ax_gamma_eeg.plot(time_plot, gamma_all_EEG[sub,:], color='blue', lw=0.8, ls=(0, (1, 1)), alpha=0.5)
        ax_gamma_opm.plot(time_plot, gamma_all_OPM[sub,:], color='blue', lw=0.5, ls=(0, (1, 1)), alpha=0.5)
    fig.tight_layout()
    plt.show()


# ERS spectograms of all subs for three selected channels for either EEG or OPM data
def plot_ers_allSubs_ind(session):
    if session == 'EEG':
        power_hand = np.load(os.path.join(data_path, 'PowerEEG', 'AllSubs_meanT_power_hand_EEG.npy'), allow_pickle=True) # trials x chs x freqs x time
        power_rest = np.load(os.path.join(data_path, 'PowerEEG', 'AllSubs_meanT_power_rest_EEG.npy'), allow_pickle=True) # trials x chs x freqs x time
        ch_names = ['FC1', 'C3', 'CP1']
    if session == 'OPM':
        power_hand = np.load(os.path.join(data_path, 'Power_OPM', 'AllSubs_meanT_power_hand_OPM.npy'), allow_pickle=True) # trials x chs x freqs x time
        power_rest = np.load(os.path.join(data_path, 'Power_OPM', 'AllSubs_meanT_power_rest_OPM.npy'), allow_pickle=True) # trials x chs x freqs x time
        ch_names = ['Sensor 2','Sensor 3','Sensor 4']
    ers = (power_hand-power_rest)/power_rest*100
    colors = ['slategrey', 'darkcyan', 'steelblue']
    fig = plt.figure(figsize=(15,20))
    time_plot = np.arange(-0.5,2.002,0.002)
    freqs = np.arange(13,120,2)
    for sub in range(6):
        for sensor in range(3):
            ax = fig.add_subplot(6,3,sub*3+sensor+1)
            im= ax.pcolormesh(time_plot, freqs,ers[sub,sensor,:,:].astype(float))
            cbar = fig.colorbar(im, ax=ax)
            if sensor == 0:
                ax.set_ylabel('Frequency [Hz]', fontsize='small')
            if sub == 2:
                ax.set_xlabel('time [s]')
            if sub == 0:
                ax.set_title(ch_names[sensor], color=colors[sensor], weight='bold')
            if sensor == 2:
                cbar.set_label('ERS [%]')
    fig.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.95, wspace=0.1, hspace=0.3)
    fig.show()
    
def plot_ers_last3():
    power_hand = np.load(os.path.join(data_path, 'PowerEEG', 'AllSubs_meanT_power_hand_EEG.npy'), allow_pickle=True) # trials x chs x freqs x time
    power_rest = np.load(os.path.join(data_path, 'PowerEEG', 'AllSubs_meanT_power_rest_EEG.npy'), allow_pickle=True) # trials x chs x freqs x time
    ch_names = ['FC1', 'C3', 'CP1']
    ers = (power_hand-power_rest)/power_rest*100
    colors = ['slategrey', 'darkcyan', 'steelblue']
    fig = plt.figure(figsize=(15,20))
    time_plot = np.arange(-0.5,2.002,0.002)
    freqs = np.arange(13,120,2)
    for sub_count,sub in enumerate([10,11,12]):
        for sensor in range(3):
            ax = fig.add_subplot(3,3,sub_count*3+sensor+1)
            im= ax.pcolormesh(time_plot, freqs,ers[sub-1,sensor,:,:].astype(float))
            cbar = fig.colorbar(im, ax=ax)
            if sensor == 0:
                ax.set_ylabel('Frequency [Hz]', fontsize='small')
            if sub == 2:
                ax.set_xlabel('time [s]')
            if sub == 0:
                ax.set_title(ch_names[sensor], color=colors[sensor], weight='bold')
            if sensor == 2:
                cbar.set_label('ERS [%]')
    fig.show()


# ERS Data of all channels of one participant
def plot_all_channels(sub, session):
    ts = 0.001
    data_path = '/Users/cntlab/Desktop/Thesis/Output/'
    if session == 'EEG':
        epochs_hand_EEG = mne.read_epochs(os.path.join(data_path, 'Data_clean', 'EEG', 'hand_clean_sess1_P'+str(sub)+'_epo.fif'), preload=True, verbose='Warning')
        power_hand = np.load(os.path.join(data_path, 'PowerEEG', 'Sub'+str(sub)+'power_hand_EEG.npy'), allow_pickle=True) # trials x chs x freqs x time
        power_rest = np.load(os.path.join(data_path, 'PowerEEG', 'Sub'+str(sub)+'power_rest_EEG.npy'), allow_pickle=True) # trials x chs x freqs x time
        ch_names = epochs_hand_EEG.ch_names
        if sub == 3 or sub == 12:
            ts = 0.002
    if session == 'OPM':
        epochs_hand_OPM = mne.read_epochs(os.path.join(data_path, 'Data_clean', 'hand_clean_sess1_P'+str(sub)+'_epo.fif'), preload=True, verbose='Warning')
        power_hand = np.load(os.path.join(data_path, 'Power_OPM', 'Sub'+str(sub)+'power_hand_OPM.npy'), allow_pickle=True) # trials x chs x freqs x time
        power_rest = np.load(os.path.join(data_path, 'Power_OPM', 'Sub'+str(sub)+'power_rest_OPM.npy'), allow_pickle=True) # trials x chs x freqs x time
        ch_names = epochs_hand_OPM.ch_names
    hand_trial_avg = np.mean(power_hand,axis=0)
    rest_trial_avg = np.mean(power_rest, axis=0)
    num_sensor_hand = hand_trial_avg.shape[0]
    num_sensor_rest = rest_trial_avg.shape[0]
    if num_sensor_hand != num_sensor_rest:
        min_sensor = min(num_sensor_rest, num_sensor_hand)
        hand_trial_avg = hand_trial_avg[0:min_sensor-1,:,:]
        rest_trial_avg = rest_trial_avg[0:min_sensor-1,:,:]
    ers = (hand_trial_avg-rest_trial_avg)/rest_trial_avg*100
    time = np.arange(-1,2+ts,ts)
    freqs = np.arange(13,120,2)
    if power_hand.shape[1] >= 15:
        max_sensor = 15
    else:
        max_sensor = ers.shape[0]
    fig_ind_ers = plt.figure(figsize = (20,15))
    for sensor in range(max_sensor):
        ax = fig_ind_ers.add_subplot(3,5,sensor+1)
        im= ax.pcolormesh(time, freqs,ers[sensor,:,:].astype(float))
        fig_ind_ers.colorbar(im, ax=ax)
        ax.set_title(ch_names[sensor])
        ax.set_xticks(np.arange(-1,2.2,0.5))
        if sensor % 5 == 0:
            ax.set_ylabel('Frequency [Hz]')
        if sensor > 9:
            ax.set_xlabel('Time [s]')
    #fig_ind_ers.suptitle('Sub '+str(sub)+': '+session+' data')
    #fig_ind_ers.tight_layout()
    fig_ind_ers.subplots_adjust(left=0.04, bottom=0.05, right=0.98, top=0.95, wspace=0.1, hspace=0.45)
    plt.show()


############################       Run following Functions for Figures          ################################

# FIGURE 1: 3 x 2 panels, showing mean beta ERS (upper panels), mean high gamma ERS (middle panel) and
#           differences in PSDs (lower panel) for OPM data (left panels) or EEG data (right panels)
# Corresponds to Figure 2 in thesis
# Plot is based on all participants
plot_ers_allSubs1ch()


# FIGURE 2: 6 x 3 panels, showing tf-spectograms of ERS values for three selected channels (horizontally)
#           and for six participants included in the analysis (vertically)
#           Data is plotted for one session
# Corresponds to Figure 5 and 7 in thesis
plot_ers_allSubs_ind('EEG')


# Figure 3: 3 x 3 panels, howing tf-spectograms of ERS values for three selected channels (horizontally)
#           and for three participants excluded from main analysis (vertically)
#           Data is plotted only for EEG session
# Corresponds to Figure 11 in thesis
plot_ers_last3()


# Figure 4: 3 x 5 panels showing tf-spectograms of ERS values of 1 participant and specific session
#           shows data of 15 sensors
# Corresponds to figures in appendix
plot_all_channels(sub=12, session='OPM')