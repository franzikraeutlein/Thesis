# Imports
import os
import mne
import numpy as np

mne.viz.set_browser_backend('matplotlib', verbose=None)

data_path = '/Users/cntlab/Desktop/Thesis/Output/Archiv/Data2'
out_path = '/Users/cntlab/Desktop/Thesis/Output/Data_clean'

# Returns idxs of epochs with data ranges above or below mean+-2*std --> Extreme epochs
def get_bad_epochs(data):
    num_sensors = data.shape[1]
    num_trials = data.shape[0]
    mins = np.min(data, axis=2)
    maxs = np.max(data, axis=2)
    range_vals = maxs-mins
    
    mean_range = np.empty(num_sensors)
    std_range = np.empty(num_sensors)
    # Exclude epochs with ranges > 80pT prior to computing mean and std
    for sensor in range(num_sensors):
        extremes = np.where(range_vals[:,sensor] > 80, 1,0)
        indices_bad_sensor = np.nonzero(extremes)[0]
        sens_range_red = np.delete(range_vals[:,sensor], indices_bad_sensor, axis=0)
        mean_sens = np.mean(sens_range_red, axis=0)
        mean_range[sensor] = mean_sens
        std_sens = np.std(sens_range_red, axis=0)
        std_range[sensor] = std_sens

    # boundaries defining inclusion criteria for data ranges of epochs; sensor-individual ranges
    # Exclude extreme epochs (exceeding mean_range +/- 2*std) 
    l_bound= mean_range - 2*std_range
    h_bound = mean_range + 2*std_range

    bads = np.empty([num_trials, num_sensors])
    indices_bad = np.zeros([num_trials, num_sensors])
    # Find indices of epochs which exceed boundaries
    for sensor in range(num_sensors):
        bads[:,sensor] = np.where(np.logical_or((range_vals[:,sensor] > h_bound[sensor]),(range_vals[:,sensor] < l_bound[sensor])), 1,0)
        indices_bad[np.nonzero(bads[:,sensor])[0],sensor]+=1
    
    # Combine Data across sensors: If at least one sensor in the data range exceede the boundaries, the epoch is excluded
    bad_epochs = np.nonzero(np.sum(indices_bad, axis=1)>0)[0]
    
    return bad_epochs


#####################################      Start Preprocessing      ################################################
sub = 11
# Load Data
raw_annotated = mne.io.Raw(os.path.join(data_path, 'raw_annotated_sess1_P'+str(sub)+'.fif'), preload=True)

# Notch filter and bandpass
raw_annotated_f = raw_annotated.copy().notch_filter(np.arange(50, 200, 50))
raw_annotated_f = raw_annotated_f.filter(l_freq=2, h_freq=200)

# Create Epochs
events, events_dict = mne.events_from_annotations(raw_annotated_f)
epochs_hand = mne.Epochs(raw_annotated_f,
            events=events,
            event_id=1,
            tmin=-1.5,
            tmax=4,
            baseline=None,
            preload=True)
epochs_rest = mne.Epochs(raw_annotated_f,
            events=events,
            event_id=2,
            tmin=-1.5,
            tmax=4,
            baseline=None,
            preload=True)
    
# CLEAN DATA
# Check if channels need to be excluded:
# If there is a messy channel, most epochs will be excluded otherwise, as for get_bad_epochs function one bad channel
# Suffices to exclude whole epoch
epochs_rest.average().plot()
    
# 1. Data Range
#### Exclude bad epochs:
hand_data = epochs_hand.get_data()*10**12 # convert to pT
rest_data = epochs_rest.get_data()*10**12 # convert to pT
    
print(hand_data.shape[0], rest_data.shape[0])

### Find bad epochs
bad_epochs_hand = get_bad_epochs(hand_data)
bad_epochs_rest = get_bad_epochs(rest_data)
print(len(bad_epochs_hand), len(bad_epochs_rest))
    
# annotations of raw filtered data 
annos_hand = np.nonzero(raw_annotated_f.annotations.description == 'Hand')[0]
annos_rest = np.nonzero(raw_annotated_f.annotations.description == 'Rest')[0]

### Change annotation description to bad
# onsets of bad hand epochs:
onset_hand_bad = annos_hand[bad_epochs_hand]
onset_rest_bad = annos_rest[bad_epochs_rest]
    
# Loop through annotations of raw filtered data and check for each annotation, whether it belongs to a
# "bad" hand or rest epoch and change annotation accordingly to "bad_Hand" or "bad_Rest"
# All remaining annotations belonging to good epochs can remain unchanged
for i in range(len(raw_annotated_f.annotations.description)):
    anno = raw_annotated_f.annotations[i]
    anno_ons = anno['onset']
    anno_dur = anno['duration']
    anno_desc = anno['description']
    if i == 0:
        print("Initializing annotation structure")
        if i in onset_hand_bad:
            annotations = mne.Annotations(onset = anno_ons,  # in seconds
                       duration = anno_dur,  # in seconds, too
                       description='bad_Hand')
        elif i in onset_rest_bad:
            annotations = mne.Annotations(onset = anno_ons,  # in seconds
                       duration = anno_dur,  # in seconds, too
                       description='bad_Rest')
        else:
            annotations = mne.Annotations(onset = anno_ons,  # in seconds
                       duration= anno_dur,  # in seconds, too
                       description=anno_desc)
    else:
        if i in onset_hand_bad:
            annotations.append(onset = anno_ons,  # in seconds
                        duration = anno_dur,  # in seconds, too
                        description='bad_Hand')
        elif i in onset_rest_bad:
            annotations.append(onset = anno_ons,  # in seconds
                        duration = anno_dur,  # in seconds, too
                        description='bad_Rest')
        else:
            annotations.append(onset = anno_ons,  # in seconds
                        duration = anno_dur,  # in seconds, too
                        description=anno_desc)

# Set annotations of raw file to go-events
raw_annotated_clean = raw_annotated_f.copy().set_annotations(annotations)

# Create Epochs
events, events_dict = mne.events_from_annotations(raw_annotated_clean)

epochs_hand_clean = mne.Epochs(raw_annotated_clean,
                    events=events,
                    event_id=1,
                    tmin=-1.5,
                    tmax=4,
                    baseline=None,
                    preload=True)

epochs_rest_clean = mne.Epochs(raw_annotated_clean,
                    events=events,
                    event_id=2,
                    tmin=-1.5,
                    tmax=4,
                    baseline=None,
                    preload=True)
    
# 2. Visual inspection
epochs_hand_clean.plot(scalings=20e-12)
epochs_rest_clean.plot(scalings=20e-12)
    
print(epochs_hand_clean.get_data(copy=False).shape[0], epochs_rest_clean.get_data(copy=False).shape[0])

# Save cleaned raw and epochs
raw_annotated_clean.save(os.path.join(out_path, 'clean_sess1_P'+str(sub)+'_raw.fif'), overwrite=True)
epochs_hand_clean.save(os.path.join(out_path, 'hand_clean_sess1_P'+str(sub)+'_epo.fif'), overwrite=True)
epochs_rest_clean.save(os.path.join(out_path, 'rest_clean_sess1_P'+str(sub)+'_epo.fif'), overwrite=True)



### Update: New Preprocessing including ICA for Subs 10-12
epochs_hand.plot(scalings=20e-12)

# ICA
ica_h = mne.preprocessing.ICA(n_components=13, method='fastica')
ica_h.fit(epochs_hand_clean)
ica_h.plot_sources(epochs_hand_clean)
# look at ICA components - do we see artifacts? If so, we can exclude them
exclude_h = [0,1,2] 
ica_h.plot_overlay(epochs_hand.average(), exclude=exclude_h) 
ratio = ica_h.get_explained_variance_ratio(epochs_hand, components=exclude_h)
print(ratio)
# and apply to data
epochs_clean2 = ica_h.apply(epochs_hand_clean, exclude=exclude_h)

epochs_clean2.plot(scalings=20e-13)

epochs_clean2.save(os.path.join(out_path, 'hand_clean1_sess1_P'+str(sub)+'_epo.fif'), overwrite=True)