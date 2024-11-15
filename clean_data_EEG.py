# Imports
import os
import mne
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.fftpack import fft
from scipy import stats
from autoreject import AutoReject
from autoreject import get_rejection_threshold
from mne.preprocessing import ICA


data_path = '/Users/cntlab/Desktop/Thesis/Output/EEG'
out_path = '/Users/cntlab/Desktop/Thesis/Output/Data_clean/EEG'
mne.viz.set_browser_backend('matplotlib', verbose=None)

# Load Data
sub = 4
raw_annotated = mne.io.Raw(os.path.join(data_path, 'raw_annotated_sess1_P'+str(sub)+'.fif'), preload=True)

# Notch filter and bandpass
raw_annotated_f = raw_annotated.copy().notch_filter(np.arange(50, 201, 50))
raw_annotated_f = raw_annotated_f.filter(l_freq=2, h_freq=200)

# Create Epochs
events, events_dict = mne.events_from_annotations(raw_annotated_f)
        
epochs = mne.Epochs(raw_annotated_f,
                    events=events,
                    event_id=[1,2],
                    tmin=-1.5,
                    tmax=3,
                    #baseline=(-1,0),
                    baseline=None,
                    preload=True)

        
# ICA
ica_h = mne.preprocessing.ICA(n_components=13, method='fastica')
ica_h.fit(epochs)
ica_h.plot_sources(epochs)
# look at ICA components - do we see artifacts? If so, we can exclude them
exclude_h = [1] 
ica_h.plot_overlay(epochs.average(), exclude=exclude_h) 
ratio = ica_h.get_explained_variance_ratio(epochs, components=exclude_h)
print(ratio)
# and apply to data
epochs_clean = ica_h.apply(epochs, exclude=exclude_h)

# Then Visual Inspection
epochs_clean.plot()

# Separate events
epochs_hand_clean = epochs_clean['1']
epochs_rest_clean = epochs_clean['2']
    
# Save Clean Epochs
epochs_hand_clean.save(os.path.join(out_path, 'hand_clean_sess1_P'+str(sub)+'_epo.fif'), overwrite=True)
epochs_rest_clean.save(os.path.join(out_path, 'rest_clean_sess1_P'+str(sub)+'_epo.fif'), overwrite=True)
    
    


