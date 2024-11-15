# Imports
import os
import pyxdf
import mne
from xdf2mne_copyRep.xdf2mne.xdf2mne import streams2raw
import json
import numpy as np
import pandas as pd
import ast

# xdf2mne developed by Jan Zefowski, available at https://github.com/jzerfowski/xdf2mne

# Running this script takes xdf data output of LSL stream and returns raw data format required for MNE analyses


data_path = '/Users/cntlab/Desktop/Thesis/Data'
out_path = '/Users/cntlab/Desktop/Thesis/Output/Data'
data_count = pd.read_excel('/Users/cntlab/Desktop/Thesis/Data2/Data_Count_OPM.xlsx', header=0)
num_subs = len(data_count.Sub)

for num,sub in enumerate(data_count.Sub):
    if sub in [5,7,9]: # excluded participants
        continue
    raw_list = list()
    num_sets = data_count.loc[num, 'Runs']
    for set in range(num_sets):
        filepath = os.path.join(data_path, 'P'+str(sub),'Hand_'+str(set+1)+'.xdf')
        streams, fileheader = pyxdf.load_xdf(filepath, dejitter_timestamps=True)
        streams = {stream["info"]["name"][0]: stream for stream in streams}
        
        stream = streams["FieldLineOPM"]
        marker_stream = streams["TaskOutput"]

        fieldData = stream['info']['desc'][0]['channels'][0]['channel'] # list of 16 element, each dict. per sensor
        raw_single = streams2raw(stream, marker_streams=[marker_stream])       
       
        raw_list.append(raw_single)

    raws = mne.concatenate_raws(raw_list)
    chs_sub = data_count.loc[num, 'Channels']
    chs_sub = ast.literal_eval(chs_sub)
    raws.rename_channels(chs_sub)

    chs = [str(item) for item in np.arange(1,16)]
    
    raws = raws.pick(chs)
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



