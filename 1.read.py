# -*- coding: utf-8 -*-
"""
Created on Sat Sep 26 22:44:53 2020

@author: REZA
"""

#%% new python code
import numpy as np

from libs.loading import loadthings



# %%
#subjects=range(1,21) #1:20
#rawdata,labels=loadthings(subjects[0])

tempD=[]
tempL=[]
Data=[]
Labels=[]
for i in range(1,20):
    Data,tempL=loadthings(i)
    Data = np.concatenate((Data, Data),axis=1)
    Labels = np.concatenate((tempL, tempL),axis=0)
    
#%%
xx=[]
yy=[]
for i in range(1,21):
    Data,Label=loadthings(i)
    xx.append(Data)
    yy.append(Label)

xx=np.reshape(xx,(64,14445*20)).T
yy=np.reshape(yy,(1,14445*20)).T   

np.save("data/Alldata", xx)

#%%
print('[info] Train/Test split and ready to run! ')



DATA_ALL = np.append(xx,yy,axis=1)
rowrank =np.random.permutation(288900)

All_of_Dataset = DATA_ALL[rowrank, :]

row=288900
#%%
tt=int(np.fix(row/10*9))

training_set   = All_of_Dataset[0:tt , 0:63];
training_label = All_of_Dataset[0:tt , 64];

test_set       = All_of_Dataset[tt+1:, 0:63];
test_label     = All_of_Dataset[tt+1:, 64];

np.save("data/training_set", training_set)
np.save("data/training_label", training_label)
np.save("data/test_set", test_set)
np.save("data/test_label", test_label)

print('[info] Everything is ready now! ')






























