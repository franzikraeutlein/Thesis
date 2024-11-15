import matplotlib.pyplot as plt
import numpy as np
import os
import mne
from mne.stats import combine_adjacency
from mne.stats import spatio_temporal_cluster_test
from scipy import stats

data_path = '/Users/cntlab/Desktop/Thesis/Output/'


##########################################     Computation  Functions    ###########################################################
def getDataTest(session, fband, sub, selected_chs=[], sfreq=1000, fmin=None, fmax=None):
    if not fband in ['Beta', 'Gamma', 'Custom']:
        print('Error: fband must be either "Beta", "Gamma" or "Custom"')
        return None,None,None
    if not session in ['EEG', 'OPM']:
        print('Error: Session must bei either "EEG" or "OPM".')
        return None,None,None
    freqs = np.arange(13,120,2) # Frequencies of power data
    if session == 'EEG':
        epochs_hand_EEG = mne.read_epochs(os.path.join(data_path, 'Data_clean', 'EEG', 'hand_clean_sess1_P'+str(sub)+'_epo.fif'), preload=True, verbose='Warning')
        ch_names = epochs_hand_EEG.ch_names
        if len(selected_chs) == 0: # Use default sensors
            selected_channels_names = np.array(['FC1', 'C3', 'CP1'])
            selected_channels_idx = np.nonzero([ch_name in selected_channels_names for ch_name in ch_names])[0]
        else:
            selected_channels_names = selected_chs
            selected_channels_idx = np.nonzero([ch_name in selected_channels_names for ch_name in ch_names])[0]
    
        power_hand = np.load(os.path.join(data_path, 'PowerEEG', 'Sub'+str(sub)+'power_hand_EEG.npy'), allow_pickle=True) # trials x chs x freqs x time
        power_rest = np.load(os.path.join(data_path, 'PowerEEG', 'Sub'+str(sub)+'power_rest_EEG.npy'), allow_pickle=True) # trials x chs x freqs x time
        if sub == 3 or sub == 12: # These two subjects were mistakenly sampled with 500Hz instead of 1000Hz
            sfreq=500
    
    elif session == 'OPM':
        if len(selected_chs) == 0: # Use default sensors
            selected_channels_names = np.array([2,3,4])
            selected_channels_idx = selected_channels_names-1
        else:
            selected_channels = selected_chs
            selected_channels_names = selected_channels-1
        power_hand = np.load(os.path.join(data_path, 'Power_OPM', 'Sub'+str(sub)+'power_hand_OPM.npy'), allow_pickle=True) # trials x chs x freqs x time
        power_rest = np.load(os.path.join(data_path, 'Power_OPM', 'Sub'+str(sub)+'power_rest_OPM.npy'), allow_pickle=True) # trials x chs x freqs x time
    if fband == 'Beta':
        fmin = 13
        fmax = 30
    elif fband == 'Gamma':
        fmin = 60
        fmax = 90
    try:
        indices_fband = np.where((freqs >= fmin)&(freqs <= fmax))[0]
    except:
        print('Error: For Custom fband fmin and fmax must be specified.')
        return None,None,None
   
    # Select data for given fband and channels
    # Power Data in time interval -1 to 2s, reduce to time window 0.5 to 1.5, chosen fband and selected channels
    if sfreq == 500:
        fband_power_hand = power_hand[:,:,indices_fband,750:-250]
        fband_power_rest = power_rest[:,:,indices_fband,750:-250]
    if sfreq == 1000:
        fband_power_hand = power_hand[:,:,indices_fband,1500:-500]
        fband_power_rest = power_rest[:,:,indices_fband,1500:-500]
    
    fband_power_hand = fband_power_hand[:,selected_channels_idx,:,:]
    fband_power_rest = fband_power_rest[:,selected_channels_idx,:,:]
    # transpose, as cluster permutation test requires channels to be the last dimension
    fband_power_hand_t = np.transpose(fband_power_hand, (0, 2, 3, 1))
    fband_power_rest_t = np.transpose(fband_power_rest, (0, 2, 3, 1))

    # X: List of both conditions
    X = [fband_power_hand_t.astype(float), fband_power_rest_t.astype(float)]
    print('Conduct Cluster-based Permutation test for Data in Time Interval 0.5-1.5s and between '+str(fmin)+' and '+str(fmax)+'Hz.')
    return X, sfreq, selected_channels_names


def getDataTest_allSubs(session, fband, sfreq=500, fmin=None, fmax=None):
    data_path = '/Users/cntlab/Desktop/Thesis/Output/'
    if not fband in ['Beta', 'Gamma', 'Custom']:
        print('Error: fband must be either "Beta", "Gamma" or "Custom"')
        return None,None,None
    if not session in ['EEG', 'OPM']:
        print('Error: Session must bei either "EEG" or "OPM".')
        return None,None,None
    freqs = np.arange(13,120,2) # Frequencies of power data
    if session == 'EEG':
        selected_channels_names = np.array(['FC1', 'C3', 'CP1'])
        selected_channels_idx = [0,1,2]
        power_hand = np.load(os.path.join(data_path, 'PowerEEG', 'AllSubs_meanT_power_hand_EEG.npy'), allow_pickle=True) # trials x chs x freqs x time
        power_rest = np.load(os.path.join(data_path, 'PowerEEG', 'AllSubs_meanT_power_rest_EEG.npy'), allow_pickle=True) # trials x chs x freqs x time
    elif session == 'OPM':
        selected_channels_names = np.array([2,3,4])
        selected_channels_idx = [0,1,2]
        power_hand = np.load(os.path.join(data_path, 'Power_OPM', 'AllSubs_meanT_power_hand_OPM.npy'), allow_pickle=True) # trials x chs x freqs x time
        power_rest = np.load(os.path.join(data_path, 'Power_OPM', 'AllSubs_meanT_power_rest_OPM.npy'), allow_pickle=True) # trials x chs x freqs x time
    if fband == 'Beta':
        fmin = 13
        fmax = 30
    elif fband == 'Gamma':
        fmin = 60
        fmax = 90
    try:
        indices_fband = np.where((freqs >= fmin)&(freqs <= fmax))[0]
    except:
        print('Error: For Custom fband fmin and fmax must be specified.')
        return None,None,None
   
    # Select data for given fband and channels
    # -----Hand
    # Power Data in time interval -0.5 to 2s, reduce to time window 0.5 to 1.5, chosen fband and selected channels
    fband_power_hand = power_hand[:,:,indices_fband,500:-250]
    fband_power_hand = fband_power_hand[:,selected_channels_idx,:,:]
    # transpose, as cluster permutation test requires channels to be the last dimension
    fband_power_hand_t = np.transpose(fband_power_hand, (0, 2, 3, 1))

    #-----Rest
    # Power Data in time interval -0.5 to 2s, reduce to time window 0.5 to 1.5, chosen fband and selected channels
    fband_power_rest = power_rest[:,:,indices_fband,500:-250]
    fband_power_rest = fband_power_rest[:,selected_channels_idx,:,:]
    # transpose, as cluster permutation test requires channels to be the last dimension
    fband_power_rest_t = np.transpose(fband_power_rest, (0, 2, 3, 1))

    # X: List of both conditions
    X = [fband_power_hand_t.astype(float), fband_power_rest_t.astype(float)]
    print('Conduct Cluster-based Permutation test for Data in Time Interval 0.5-1.5s and between '+str(fmin)+' and '+str(fmax)+'Hz.')
    return X, sfreq, selected_channels_names


def get_adj_matrix(X, adj_chs=None):
    # Combine Adjacency
    n_times = X[0].shape[2]
    n_freqs = X[0].shape[1]
    # adjacency of default channels, if other channels are selected, adj matrix must be provided
    if isinstance(adj_chs, (np.ndarray)):
        adj_m = adj_chs
    else:
        adj_m = np.array([[0,1,0],
                          [1,0,1],
                          [0,1,0]])
    adj_matrix = combine_adjacency(
        n_freqs,  # regular lattice adjacency for times
        n_times,  # regular lattice adjacency for freqs
        adj_m)  
    return adj_matrix

# Identify significant clusters
def sign_clusters(T_obs, clusters, cluster_p_values, fband, sfreq, selected_channels, fmin=None, fmax=None):
    freqs = np.arange(13,120,2)
    # Find significant clusters
    ts = 1/sfreq
    s_c = np.where(cluster_p_values < 0.05)[0]
    if len(s_c)==0:
        print('There were no significant clusters.')
        return
    else:
        c_based_time = np.arange(0.5,1.5+ts,ts)
        if fband == 'Beta':
            indices_fband = np.where((freqs >= 13)&(freqs <= 30))[0]
            freqs_fband = freqs[indices_fband]
        if fband == 'Gamma':
            indices_fband = np.where((freqs >= 60)&(freqs<=90))[0]
            freqs_fband = freqs[indices_fband]
        if fband == 'Custom':
            try:
                indices_fband = np.where((freqs >= fmin)&(freqs <= fmax))[0]
                freqs_fband = freqs[indices_fband]
            except:
                print('For Custom fband fmin and fmax must be specified.')
                return
        for num_cluster in range(len(s_c)):
            c_idx = s_c[num_cluster]
            cluster_points = clusters[c_idx]
            p_val = cluster_p_values[s_c[num_cluster]]
            cluster_T_values = T_obs[cluster_points]
            freq_index = cluster_points[0]
            freqs_cluster = freqs_fband[freq_index]
            time_index = cluster_points[1]
            time_cluster = c_based_time[time_index]
            ch_index = cluster_points[2]
            ch_cluster = selected_channels[ch_index]
            involved_freqs = np.unique(freqs_cluster)
            involved_chs = np.unique(ch_cluster)
            
            print('Results Cluster '+str(num_cluster+1)+' of '+str(len(s_c))+':')
            print('Time ranges from '+str(round(np.min(time_cluster),2))+'s to '+str(round(np.max(time_cluster),2))+'s')
            print('Involved Frequencies range from '+str(involved_freqs[0])+'Hz to '+str(involved_freqs[-1])+'Hz')
            print('Involved Channels are: '+str(involved_chs))
            print('Summed t-value: '+str(cluster_T_values.sum()))
            print('p-value: '+str(p_val))
        return involved_freqs

# Run Cluster Permutation Test and print Results
# sub either subject number or 'all'
def cluster_based_permutation_fband(session, fband, sub, selected_chs=[], sfreq=1000, fmin=None, fmax=None, adj_chs=None):
    if sub == 'all':
        X,sampling_freq,sel_chs = getDataTest_allSubs(session=session, fband=fband, sfreq=500, fmin=fmin, fmax=fmax)
    else:
        X,sampling_freq,sel_chs = getDataTest(session=session, fband=fband, sub=sub, selected_chs=selected_chs, sfreq=sfreq, fmin=fmin, fmax=fmax)
    if not isinstance(X, (list)):
        return
    adj_m = get_adj_matrix(X=X, adj_chs=adj_chs)
    # Run Test
    T_obs, clusters, cluster_p_values, H0 = spatio_temporal_cluster_test(X, n_permutations=1000, tail=1, out_type='indices', adjacency=adj_m)
    involved_freqs = sign_clusters(T_obs=T_obs, clusters=clusters, cluster_p_values=cluster_p_values, fband=fband, sfreq=sampling_freq, selected_channels=sel_chs, fmin=fmin, fmax=fmax)
    return involved_freqs

##########################################     Plotting  Functions    ###########################################################

def plot_cbp_results_allSubs(sub = [], power_hand=None, power_rest=None, session='OPM', sfreq=1000, fmin=[60,13],fmax=[90,13]):
    found_cluster = 2
    freqs = np.arange(13,120,2)
    indices_fband_gamma = np.where((freqs >= fmin[0])&(freqs<=fmax[0]))[0]
    if len(indices_fband_gamma) == 0:
        found_cluster = -1
    indices_fband_beta = np.where((freqs >= fmin[1])&(freqs<=fmax[1]))[0]
    if len(indices_fband_beta) == 0:
        if found_cluster == 2:
            found_cluster = -2
        if found_cluster == -1:
            print('This plotting function is not suitable for this subject, as no significant clusters were found.')
            return
    if session == 'OPM':
        title = ' OPM Data'
        # Load stacked power data of all Subs
        if len(sub) == 0:
            power_hand = np.load(os.path.join(data_path, 'Power_OPM', 'AllSubs_indT_power_hand_OPM.npy'), allow_pickle=True)
            power_rest= np.load(os.path.join(data_path, 'Power_OPM', 'AllSubs_indT_power_rest_OPM.npy'), allow_pickle=True)
    if session == 'EEG':
        title = ' EEG Data'
        # Load stacked power data of all Subs
        if len(sub) == 0:
            power_hand = np.load(os.path.join(data_path, 'PowerEEG', 'AllSubs_indT_power_hand_EEG.npy'), allow_pickle=True)
            power_rest= np.load(os.path.join(data_path, 'PowerEEG', 'AllSubs_indT_power_rest_EEG.npy'), allow_pickle=True)
    fig = plt.figure(figsize=(20,15))
    if found_cluster == -1: # kein gamma Cluster
        fig = timeSeries_sign_power(power_hand,power_rest,'beta', indices_fband_beta,session,sfreq,found_cluster, fig)
    if found_cluster == -2: # kein beta Cluster
        fig = timeSeries_sign_power(power_hand,power_rest,'gamma', indices_fband_gamma,session,sfreq,found_cluster,fig)
    if found_cluster == 2: # beide Cluster da
        fig = timeSeries_sign_power(power_hand,power_rest,'beta', indices_fband_beta,session,sfreq,found_cluster, fig)
        fig = timeSeries_sign_power(power_hand,power_rest,'gamma', indices_fband_gamma,session,sfreq,found_cluster,fig)
    mean_hand = np.mean(power_hand,axis=0)
    mean_rest = np.mean(power_rest, axis=0)
    ers = (mean_hand-mean_rest)/mean_rest*100
    fig = plot_power(ers, sfreq, session, found_cluster, fig)
    if len(sub) == 0:
        fig_title = 'Cluster Based Analysis of'+title
    else:
        fig_title = 'Cluster Based Analysis of'+title+', Sub '+str(sub[0])
    fig.suptitle(fig_title)
    plt.show()
    
def timeSeries_sign_power(power_hand, power_rest, fband, indices_fband, session, sfreq, found_cluster,fig):
    ts = 1/sfreq
    time_plot = np.arange(-0.5,2+ts,ts)
    if fband == 'gamma':
        pos = [1,2,3]
    if fband == 'beta':
        if found_cluster == -1:
            pos = [4,5,6]
        else:
            pos = [7,8,9]
    if session == 'OPM':
        ch_names = [2,3,4]
        title = 'Sensor '
    if session == 'EEG': 
        ch_names = ['FC1', 'C3', 'CP1']
        title = ''
    num_chs = power_hand.shape[1]
    fband_power_hand = power_hand[:,:,indices_fband,:] # -0.5 bis 2
    fband_power_rest = power_rest[:,:,indices_fband,:]

    #Average across freqs in fband
    mean_hand_fs = np.mean(fband_power_hand, axis=2)
    mean_rest_fs = np.mean(fband_power_rest, axis=2)
    # Average across trials
    sem_hand = stats.sem(mean_hand_fs.astype(float), axis=0)
    mean_hand = np.mean(mean_hand_fs, axis=0)
    sem_rest = stats.sem(mean_rest_fs.astype(float), axis=0)
    mean_rest = np.mean(mean_rest_fs, axis=0)

    for sensor in range(num_chs):
        if found_cluster != 2:
            ax = fig.add_subplot(2,3,pos[sensor])
        else:
            ax = fig.add_subplot(3,3,pos[sensor])
        ax.plot(time_plot, mean_hand[sensor,:], color='green', label='Movement')
        ax.plot(time_plot, mean_rest[sensor,:], color='red', label='Rest')
        ax.fill_between(time_plot, mean_hand[sensor,:].astype(float)+1.96*sem_hand[sensor,:], mean_hand[sensor,:].astype(float)-1.96*sem_hand[sensor,:], color='green', alpha=0.25)
        ax.fill_between(time_plot, mean_rest[sensor,:].astype(float)+1.96*sem_rest[sensor,:], mean_rest[sensor,:].astype(float)-1.96*sem_rest[sensor,:], color='red', alpha=0.25)
        if fband == 'beta':
            ax.set_xlabel('time [s]')
        if sensor == 0:
            ax.set_ylabel('Relative Power')
        if fband == 'gamma':
            ax.set_title(title+str(ch_names[sensor]))
        ax.legend(loc=2)
    return fig

def plot_power(ers, sfreq, session,found_cluster, fig):
    num_chs = ers.shape[0]
    ts = 1/sfreq
    time_plot = np.arange(-0.5,2+ts,ts)
    freqs = np.arange(13,120,2)
    if session == 'OPM': 
        ch_names = [2,3,4]
        title = 'Sensor '
    if session == 'EEG':
        ch_names = ['FC1', 'C3', 'CP1']
        title = ''
    if found_cluster == -1: # kein gamma cluster
        pos = [1,2,3]
    if found_cluster == -2 or found_cluster==2: # kein beta cluster
        pos = [4,5,6]
    for sensor in range(num_chs):
        if found_cluster == -1 or found_cluster == -2:
            ax = fig.add_subplot(2,3,pos[sensor])
        else:
            ax = fig.add_subplot(3,3,pos[sensor])
        im= ax.pcolormesh(time_plot, freqs,ers[sensor,:,:].astype(float))
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label('ERS [%]')
        #ax.set_xlabel('time [s]')
        if sensor == 0:
            ax.set_ylabel('Frequency [Hz]')
        if found_cluster == -1:
            ax.set_title(title+str(ch_names[sensor]))
    return fig


# Recreate Figure for specific Subject
def plot_cbp_results(sub,session,fmin,fmax):
    sfreq=1000
    if session == 'EEG':
        epochs_hand_EEG = mne.read_epochs(os.path.join(data_path, 'Data_clean', 'EEG', 'hand_clean_sess1_P'+str(sub)+'_epo.fif'), preload=True, verbose='Warning')
        power_hand = np.load(os.path.join(data_path, 'PowerEEG', 'Sub'+str(sub)+'power_hand_EEG.npy'), allow_pickle=True) # trials x chs x freqs x time
        power_rest = np.load(os.path.join(data_path, 'PowerEEG', 'Sub'+str(sub)+'power_rest_EEG.npy'), allow_pickle=True) # trials x chs x freqs x time
        ch_names = epochs_hand_EEG.ch_names
        selected_channels_eeg = np.array(['FC1', 'C3', 'CP1'])
        selected_channels = np.nonzero([ch_name in selected_channels_eeg for ch_name in ch_names])[0]
        if sub == 3 or sub==12:
            sfreq = 500
        
    if session == 'OPM':
        power_hand = np.load(os.path.join(data_path, 'Power_OPM', 'Sub'+str(sub)+'power_hand_OPM.npy'), allow_pickle=True) # trials x chs x freqs x time
        power_rest = np.load(os.path.join(data_path, 'Power_OPM', 'Sub'+str(sub)+'power_rest_OPM.npy'), allow_pickle=True) # trials x chs x freqs x time
        selected_channels = np.array([1,2,3])

    # reduce power (-1 bis 2s) to -0.5 bis 2s und auf nur 3 channels
    samples_5ms = int(0.5*sfreq)
    power_hand_red = power_hand[:,selected_channels,:,samples_5ms:]
    power_rest_red = power_rest[:,selected_channels,:,samples_5ms:]
    
    plot_cbp_results_allSubs(sub = [sub], power_hand=power_hand_red, power_rest=power_rest_red,session=session, sfreq=sfreq, fmin=fmin,fmax=fmax)
    

def cluster_based_permutation(session, sub, sfreq=1000, fmin=None, fmax=None):
    involved_freqs_beta = cluster_based_permutation_fband(session=session, fband='Beta', sub=sub, sfreq=sfreq, selected_chs=[], fmin=None, fmax=None, adj_chs=None)
    involved_freqs_gamma = cluster_based_permutation_fband(session=session, fband='Gamma', sub=sub, sfreq=sfreq, selected_chs=[], fmin=None, fmax=None, adj_chs=None)
    print('Creating Figure...')
    if isinstance(involved_freqs_beta, np.ndarray):
        fmin_beta = min(involved_freqs_beta)
        fmax_beta = max(involved_freqs_beta)
    else:
        fmin_beta,fmax_beta = 0,0
    if isinstance(involved_freqs_gamma, np.ndarray):
        fmin_gamma = min(involved_freqs_gamma)
        fmax_gamma = max(involved_freqs_gamma)
    else:
        fmin_gamma, fmax_gamma = 0,0
    if sub == 'all':
        plot_cbp_results_allSubs(session=session, sfreq=500, fmin=[fmin_gamma,fmin_beta],fmax=[fmax_gamma,fmax_beta])
    else:
        plot_cbp_results(sub=sub,session=session,fmin=[fmin_gamma,fmin_beta],fmax=[fmax_gamma,fmax_beta])
    return

############################################ Run #######################################################################
# Just print results of Cluster-Based-Permutation Test: 
# ---> Use cluster_based_permutation_fband(session, fband, sub, fmin=None, fmax=None)
# session: 'EEG' or 'OPM'
# fband: 'Beta', 'Gamma', or 'Custom', for 'Custom' specify fmin and fmax
# sub: Either specific subject number, i.e. 1 or 'all' for Cluster Test based on trials of all Subjects stacked together
#involved_freqs = cluster_based_permutation_fband(session, fband, sub)

# Print results and plot of Cluster-Permutation test for Beta and Gamma frequency Band
# ----> Use cluster_based_permutation(session, sub)
# session: 'EEG' or 'OPM'
# sub: Either specific subject number, i.e. 1 or 'all' for Cluster Test based on means of all Subjects stacked together

cluster_based_permutation(session='EEG', sub='all')