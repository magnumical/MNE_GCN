# -*- coding: utf-8 -*-
"""
Created on Sun Sep 27 11:30:02 2020

@author: REZA
"""


from libs import models, graph, coarsening,GCN_Model,DenseGCN_Model

#%%
import numpy as np

from scipy import sparse
from tensorflow.python.framework import ops

import tensorflow as tf
# Clear all the stack and use GPU resources as much as possible
ops.reset_default_graph()
config= tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess=tf.compat.v1.Session(config=config)


#%%

train_data= np.load('data/training_set.npy').astype('float32')
train_labels= np.load('data/training_label.npy').astype('float32')

test_data= np.load('data/test_set.npy').astype('float32')
test_labels= np.load('data/test_label.npy').astype('float32')

print('==============> Data read!')
  
#%%
test_labels=test_labels.reshape(test_labels.shape[0],)
train_labels=train_labels.reshape(train_labels.shape[0],)
#train_data=train_data.reshape(64,967680)
#test_data=test_data.reshape(64,107520)

#%%
DIR='files/'
Adjacency_Matrix = np.load('matrices/Adjacency_Matrix.npy').astype('float32')
Adjacency_Matrix = sparse.csr_matrix(Adjacency_Matrix)
print('==============> Adjancy matrix read!')


graphs, perm = coarsening.coarsen(Adjacency_Matrix, levels=5, self_connections=False)
X_train = coarsening.perm_data(train_data, perm)
X_test  = coarsening.perm_data(test_data,  perm)
print('==============>coarsening done!')


#%%


L = [graph.laplacian(Adjacency_Matrix, normalized=True) for Adjacency_Matrix in graphs]
print('==============>laplacian obtained!')
graph.plot_spectrum(L)

#%% Hyper-parameters
params = dict()
params['dir_name']       = 'folder1'

params['num_epochs']     = 100
params['batch_size']     = 1024
params['eval_frequency'] = 100

# Building blocks.
params['filter'] = 'chebyshev5'
params['brelu']  = 'b2relu'
params['pool']   = 'mpool1'

# Architecture.
params['F'] = [16, 32, 64, 128, 256, 512]  # Number of graph convolutional filters.
params['K'] = [2, 2, 2, 2, 2, 2]           # Polynomial orders.
params['p'] = [2, 2, 2, 2, 2, 2]           # Pooling sizes.
params['M'] = [2]                          # Output dimensionality of fully connected layers.

# Optimization.
params['regularization'] = 0.000001  # L2 regularization
params['dropout']        = 0.50      # Dropout rate
params['learning_rate']  = 0.000001  # Learning rate
params['decay_rate']     = 1         # Learning rate Decay == 1 means no Decay
params['momentum']       = 0         # momentum == 0 means Use Adam Optimizer
params['decay_steps']    = np.shape(train_data)[0] / params['batch_size']
print('==============>parameters  selected!')

#%%
model = models.cgcnn(L, **params)
#model = GCN_Model.cgcnn(L, **params)
#model=DenseGCN_Model.cgcnn(L, **params)

print('==============>model  established!')
accuracy, loss, t_step = model.fit(X_train, train_labels, X_test, test_labels)

print('==============>model  fitted!')



