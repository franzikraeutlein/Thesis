# Imports
import os
import pyxdf
import mne
import json
import numpy as np
import pandas as pd
from typing import List, Callable

# Running this script takes xdf data output of LSL stream and returns raw data format required for MNE analyses


#######################        Functions copied from xdf2mne (Jan Zerfowski) and adapted to EEG        ############################
# xdf2mne available at https://github.com/jzerfowski/xdf2mne
CH_TYPE_DEFAULT = 'misc'

from enum import Enum
class Unit_Factor(Enum):
    """
    Represent common orders of magnitude and their factors relative to SI unit Tesla
    """
    ONE = 1

    T = 1  # Tesla
    mT = 1E3  # milliTesla
    uT = 1E6  # microTesla
    nT = 1E9  # nanoTesla
    pT = 1E12  # picoTesla
    fT = 1E15  # femtoTesla

    V = 1  # Volt
    mV = 1E3  # milliVolt
    uV = 1E6 # microvolts

    @classmethod
    def _missing_(cls, value):
        return cls.ONE

def ch_type_transform_default(type_=None, default_type=CH_TYPE_DEFAULT):
    if default_type is None or default_type not in mne.io.get_channel_type_constants():
        default_type = CH_TYPE_DEFAULT

    if type_ is None:
        return default_type

    type_ = str(type_).lower()
    return type_ if type_ in mne.io.get_channel_type_constants() else default_type


def streams2raw(data_stream: dict, marker_streams: List[dict] = None, ch_type_t: Callable = ch_type_transform_default) -> mne.io.RawArray:
    raw, t_original = stream2raw(data_stream, ch_type_t=ch_type_t)
    raw_add_annotations(raw, t_reference=t_original, marker_streams=marker_streams)
    return raw


def stream2raw(stream, ch_type_t=None):
 # Extract information from stream
    t_original = stream['time_stamps']
    sfreq = float(stream['info']['nominal_srate'][0])

    if ch_type_t is None:
        def ch_type_transform(type_=None):
            return ch_type_transform_default('EEG', stream['info']['type'][0])
        ch_type_t = ch_type_transform()

    ## Prepare info structure
    ch_names, ch_types, ch_units = get_ch_info(stream['info'], ch_type_t)
    info = mne.create_info(ch_names, sfreq, ch_types)

    # Prepare data by bringing units into SI units
    data = np.array(stream['time_series']).T  # Transpose to make shape (n_channels, n_times)
    unit_factors = np.array([1/unit_fac.value for unit_fac in ch_units])
    data *= unit_factors[:, np.newaxis]  # Add axis for broadcasting

    raw = mne.io.RawArray(data=data, info=info)
    # raw._t_original = t_original

    return raw, t_original

def raw_add_annotations(raw: mne.io.RawArray, t_reference, marker_streams=None) -> mne.io.RawArray:
    if marker_streams:
        for stream in marker_streams:
            annotations = marker_stream2annotations(stream, t_reference=t_reference)
            raw.set_annotations(raw.annotations + annotations)

    return raw



def get_ch_info(info, ch_type_t):
    ch_names = []
    ch_types = []
    ch_units = []

    channel_count = int(info['channel_count'][0])

    if 'desc' in info:
        desc = info['desc'][0]
        if 'channels' in desc and 'channel' in desc['channels'][0]:
            for ch_info_idx, channel in enumerate(desc['channels'][0]['channel']):
                if 'label' in channel:
                    ch_label = channel['label'][0]
                    ch_names.append(ch_label)

                    if 'type' in channel:
                        if channel['type'][0] == 'EEG':
                            ch_types.append(channel['type'][0].lower())
                        elif channel['type'][0] == 'ACC':
                            ch_types.append('misc')

                    if 'unit' in channel:
                        if channel['unit'][0] == 'microvolts':
                            unit = 'uV'
                        ch_units.append(Unit_Factor[unit])
                    else:
                        ch_units.append(Unit_Factor.ONE)

                else:
                    break
                    # If a channel has no label, the "channels" structure seems corrupted and we use fallback to defaults

    return ch_names, ch_types, ch_units


def raw_add_annotations(raw: mne.io.RawArray, t_reference, marker_streams=None) -> mne.io.RawArray:
    if marker_streams:
        for stream in marker_streams:
            annotations = marker_stream2annotations(stream, t_reference=t_reference)
            raw.set_annotations(raw.annotations + annotations)

    return raw


def marker_stream2annotations(marker_stream, t_reference, t_duration=0.0):
    """
    Extract the events from a marker stream and create an mne.Annotations object
    This method requires an array or integer t_reference in relation to which "t=0" is determined.
    Expects the marker_stream and t_reference timestamp(s) to be from synced clocks.
    If there are multiple channels in marker_stream, they are concatenated with using '/'
    This allows use of epoch subselectors in mne
    https://mne.tools/dev/auto_tutorials/epochs/10_epochs_overview.html#tut-section-subselect-epochs
    :param marker_stream: A marker stream from an xdf-file
    :param t_reference: A single timestamp or an array of timestamps determine t=0
    :param t_duration: Default duration of the annotations. Could be extended in the future
    :return: mne.Annotations object containing all markers, event_id dictionary compatible with mne
    """
    # Extract channel 0 from the marker stream since mne can't handle more
    # markers = np.array(marker_stream['time_series'])[:, 0]
    markers = np.array(marker_stream['time_series']).astype(str)
    marker_strings = ['/'.join(markers[i]) for i in range(len(markers))]
    markers_t = marker_stream['time_stamps']-np.min(t_reference)

    # event_id = _get_event_id(marker_stream)
    annotations = mne.Annotations(onset=markers_t, duration=[t_duration] * len(markers_t), description=marker_strings)
    return annotations



####################################        Start Preprocessing         ###################################################

data_path = '/Users/cntlab/Desktop/Thesis/Data2/EEG'
out_path = '/Users/cntlab/Desktop/Thesis/Output/EEG'
data_count = pd.read_excel('/Users/cntlab/Desktop/Thesis/Data2/EEG/Data_Count_EEG.xlsx', header=0)
num_subs = len(data_count.Sub)
for num,sub in enumerate(data_count.Sub):
    raw_list = list()
    num_sets = data_count.loc[num, 'Runs']
    for set in range(num_sets):
        fname = os.path.join(data_path, 'P'+str(sub), 'Hand_0'+str(set+1)+'.xdf')
        streams, fileheader = pyxdf.load_xdf(fname, dejitter_timestamps=True)
        streams = {stream["info"]["name"][0]: stream for stream in streams}
        
        try: # unterschiedlich, je nachdem welcher Amplifier benutzt wurde
            stream = streams["LiveAmpSN-054211-0207"]
        except:
            stream = streams["LiveAmpSN-054208-0183"]
            
        marker_stream = streams["TaskOutput"]

        t_original = stream['time_stamps']
        sfreq = float(stream['info']['nominal_srate'][0])

        ch_type_t = None
        raw, t_original = stream2raw(stream, ch_type_t=ch_type_t) # Raw data with values in microvolts
        annotations = marker_stream2annotations(marker_stream, t_reference=t_original)
        raw_single = raw.set_annotations(raw.annotations + annotations)
        raw_list.append(raw_single)

    raws = mne.concatenate_raws(raw_list)
    
    num_of_trials=240
    events_onset = np.zeros([2,num_of_trials])
    hand_count, rest_count = 0,0
    for count, annotation in enumerate(raws.annotations):
        try:
            annos = json.loads(annotation["description"])['text']
        
            if annos == None:
                continue
            onset = annotation['onset']
            if 'Hand' in annos:
                events_onset[0,hand_count] = onset
                hand_count +=1
            if 'Rest' in annos:
                events_onset[1,rest_count] = onset
                rest_count +=1
        except:
            print('check description of iteration :'+str(count)+':')
            print(annotation)
            continue
    
    # if subject had less than 240 trials, events_onset is filled up with zeros - get rid of them
    hand_events = events_onset[0,np.nonzero(events_onset[0,:])[0]]
    rest_events = events_onset[1,np.nonzero(events_onset[1,:])[0]]

    #description_events = np.repeat(['Hand', 'Tongue', 'Rest'], num_of_trials)
    description_hand = np.repeat('Hand', len(hand_events))
    description_rest = np.repeat('Rest', len(rest_events))
    description_events = np.concatenate([description_hand, description_rest])
    onsets_flat = np.concatenate([hand_events, rest_events])        
    for i in range(len(onsets_flat)):
        if i == 0:
            print("Initializing annotation structure")
            annotations = mne.Annotations(onsets_flat[i],  # in seconds
                       duration=4,  # in seconds, too
                       description=description_events[i])
        else:
            annotations.append(onsets_flat[i],  # in seconds
                duration=4,  # in seconds, too
                   description=description_events[i])

# Set annotations of raw file to go-events
    raw_annotated = raws.set_annotations(annotations)
    
    # Save raw annotated
    raw_annotated.save(os.path.join(out_path, 'raw_annotated_sess1_P'+str(sub)+'.fif'), overwrite=True)






