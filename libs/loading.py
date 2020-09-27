# -*- coding: utf-8 -*-


from mne.datasets import eegbci
from mne.io import concatenate_raws,read_raw_edf
from mne.channels import make_standard_montage
from mne import Epochs,pick_types,events_from_annotations
import numpy as np


def loadthings(subject):
    runs=[6,10,14]
    raw_fnames=eegbci.load_data(subject,runs)
    raw=concatenate_raws([read_raw_edf(f,preload=True) for f in raw_fnames])
    

    raw_fnames=eegbci.load_data(subject,runs)
    raw=concatenate_raws([read_raw_edf(f,preload=True) for f in raw_fnames])
    eegbci.standardize(raw)
    #create 10-05 system
    montage=make_standard_montage('standard_1005')
    raw.set_montage(montage)
    raw.filter(7,13)
    events,_=events_from_annotations(raw,event_id=dict(T1=0,T2=1))
#0: left, 1:right
    picks=pick_types(raw.info,meg=False,eeg=True,stim=False,eog=False,exclude='bads')
    
    tmin,tmax=-1,4
    epochs=Epochs(raw,events,None,tmin,tmax,proj=True,picks=picks,
                 baseline=None,preload=True)
    #epochs_train=epochs.copy().crop(0,2)
    epochs_train=epochs.copy().crop(0,2)
    labels=epochs.events[:,-1]
    
    # %%
    epochs_train_data=epochs_train.get_data()
    labels=epochs_train.events[:,-1]
    
    ad=np.array(epochs_train_data)
    All_Data = ad.reshape(64,epochs_train_data.shape[2]*epochs_train_data.shape[0])
    
    labels = labels.reshape(45,1)
    
    extended=[]
    for i in range (321):
        extended.append(labels)
    extended = np.array(extended)
    extended = extended.reshape(45,321)
    extended = extended.T
    Labels = extended
    #np.savetxt("pythondata/extended.csv", extended)
    
    row, column = Labels.shape
    Labels = Labels.reshape(1,row*column)
    Labels=Labels.T    

    return All_Data,Labels










