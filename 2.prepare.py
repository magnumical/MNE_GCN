# -*- coding: utf-8 -*-
"""
Created on Sun Sep 27 11:19:15 2020

@author: REZA
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Aug 27 10:29:14 2020

@author: REZA
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
import csv

#%%

Alldata = np.load('data/Alldata.npy').astype('float32')
print('[info] All data of 64 electrodes are here!')

#%%Normalize
xx= np.array(Alldata)
NormalizedAll = xx - xx.min();
NormalizedAll = NormalizedAll / xx.max();
NormalizedAll=NormalizedAll.reshape(64,Alldata.shape[0])
print('[info]  Normalizezed')

#%%Covariance
print('[info] Calculating Covariance matrix')
covariance_matrix = np.cov(NormalizedAll);
print('[info] covariance of Normalized/Standardize data is calculated')
np.save("matrices/covariance", covariance_matrix)

plt.style.use('seaborn-poster')
plt.imshow(covariance_matrix,extent=[0,64, 0, 64],cmap='viridis')

#%% Pearson matrix and its ABS
print('[info] Calculating Pearson matrix')
Pearson_matrix= np.corrcoef(NormalizedAll)
np.savetxt("matrices/Pearson_matrix.csv", Pearson_matrix)
print('[info] Pearson matrix of Normalized/Standardize data is calculated')

print('[info] Calculating Absolute Pearson matrix')
Absolute_Pearson_matrix = abs(Pearson_matrix);
np.save("matrices/Absolute_Pearson_matrix", Absolute_Pearson_matrix)

print('[info] Absolute Pearson matrix Calculated')

plt.figure()
plt.imshow(Pearson_matrix,extent=[0,64, 0, 64],cmap='viridis')

plt.figure()
plt.imshow(Absolute_Pearson_matrix,extent=[0,64, 0, 64],cmap='viridis')

#%% Adjacency Matrix
print('[info] Calculating Adjacency Matrix')
Eye_Matrix = np.eye(64, 64);
Adjacency_Matrix = Absolute_Pearson_matrix - Eye_Matrix;

np.save("matrices/Adjacency_Matrix", Adjacency_Matrix)

print('[info] Adjacency Matrix is Calculated')

plt.figure()
plt.imshow(Adjacency_Matrix,extent=[0,64,0, 64],cmap='viridis')



#%% Degree Matrix
print('[info] Calculating Degree Matrix')
diagonal_vector = np.sum(Adjacency_Matrix,axis=1)
Degree_Matrix = np.diag(diagonal_vector)
np.savetxt("matrices/Degree_Matrix.csv", Degree_Matrix)
print('[info] Degree Matrix Calculated')

plt.figure()
plt.imshow(Degree_Matrix,extent=[0,64,0, 64],cmap='viridis')

#%% Laplacian Matrix

print('[info] Calculating Laplacian Matrix')
Laplacian_Matrix = Degree_Matrix - Adjacency_Matrix;
np.savetxt("matrices/Laplacian_Matrix.csv", Laplacian_Matrix)
print('[info] Laplacian Matrix Calculated ')

plt.figure()
plt.imshow(Laplacian_Matrix,extent=[0,64,0, 64],cmap='viridis')




















