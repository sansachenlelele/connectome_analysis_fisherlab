#!/usr/bin/env python
# coding: utf-8

# June 26th 2024  
# Make recurrent connectivity matrices for Delta7s. Note that the matrices have presynaptic neurons on rows and postsynaptic on columns.

# In[1]:


import sys, os, glob
import datetime, time

import numpy as np
from numpy import linalg as nla
#import scipy.linalg as sla
import numpy.random as nrd

# import scipy.stats as sst

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd

import scipy.io as sio
from scipy.stats import wilcoxon
from scipy.stats import binom

from collections import defaultdict

gen_fns_dir = os.path.abspath('../shared')
sys.path.append(gen_fns_dir)
from general_file_fns import load_pickle_file, save_pickle_file

curr_date=datetime.datetime.now().strftime('%Y_%m_%d')+'_'
#sd=int((time.time()%1)*(2**31))
# rng = nrd.default_rng(sd)
# print('Seed= ',sd)


# In[2]:


data_path = '../../results/2024_06_19/delta7_conn_with_normalized_weights/'
fname = 'delta7_delta7_connection_with_normalized_weights.csv'
conn_data = pd.read_csv(data_path + fname)


# In[3]:


conn_data


# In[4]:


# Get list of subtypes
subtype_list = sorted(conn_data['instance_pre'].unique())
print('Pre subtypes same as post subtypes ', sorted(conn_data['instance_post'].unique())==subtype_list)
print(subtype_list)
nsubtypes = len(subtype_list)

# Make a list of all the neurons labeled by subtype and id
pre_neuron_list = conn_data[['instance_pre', 'bodyId_pre']].apply(tuple, axis=1).unique()
post_neuron_list = conn_data[['instance_post', 'bodyId_post']].apply(tuple, axis=1).unique()
neuron_list = sorted(pre_neuron_list)
print('Same neurons in pre and postsynaptic list ', neuron_list == sorted(post_neuron_list))
print('Neuron name subtypes are same as subtype list ', np.all(pd.unique([x for x, _ in neuron_list])==subtype_list))

# Get neuron indices corresponding to each subtype
subtype_idx = defaultdict(list)
for i, nrn in enumerate(neuron_list):
    subtype_idx[nrn[0]].append(i)

# And boundaries, to help in plotting
subtype_boundaries = []
for x, y in zip(subtype_list[:-1], subtype_list[1:]):
    subtype_boundaries.append((subtype_idx[x][-1] + subtype_idx[y][0])/2.)


# In[5]:


tmp = conn_data[(conn_data['bodyId_pre']==5813061383) & (conn_data['bodyId_post']==5813048042)]


# In[6]:


tmp['norm_weight']


# In[7]:


conn_data


# In[8]:


# Make the recurrent connectivity matrix ordered according to this neuron list
raw_conn_matrix = np.zeros((len(neuron_list), len(neuron_list)))
norm_conn_matrix = np.zeros_like(raw_conn_matrix)

for i, pre in enumerate(neuron_list):
    for j, post in enumerate(neuron_list):
#        tmp = conn_data[(conn_data['bodyId_pre']==pre[1]) & (conn_data['bodyId_post']==post[1])]['weight']
        tmp = conn_data[(conn_data['bodyId_pre']==pre[1]) & (conn_data['bodyId_post']==post[1])]
        if len(tmp):
            # Earlier version stores a list of duplicate connections to check with Tianhao later.
            raw_conn_matrix[i,j] = tmp['weight'].sum()
            norm_conn_matrix[i,j] = tmp['norm_weight'].sum()


# In[11]:


# Plot
fig, ax = plt.subplots(1,2,figsize=(9,4))
sns.heatmap(raw_conn_matrix, cmap="rocket", ax=ax[0])
ax[0].invert_yaxis()
p = ax[1].imshow(raw_conn_matrix, aspect='auto', interpolation='nearest', origin='lower', cmap='hot')
fig.colorbar(p, ax=ax[1])

fig, ax = plt.subplots(1,2,figsize=(9,4))
sns.heatmap(norm_conn_matrix, cmap="rocket", ax=ax[0])
ax[0].invert_yaxis()
p = ax[1].imshow(norm_conn_matrix, aspect='auto', interpolation='nearest', origin='lower', cmap='hot')
fig.colorbar(p, ax=ax[1])


# In[10]:


to_save = {'subtype_list' : subtype_list, 'neuron_list' : neuron_list, 'subtype_idx' : subtype_idx,
           'subtype_boundaries' : subtype_boundaries, 'raw_conn_matrix' : raw_conn_matrix, 
           'norm_conn_matrix' : norm_conn_matrix}
data_path = '../../results/2024_06_26/delta7_rec_matrices/'
fname = 'delta7_delta7_connections_rec_matrices_with_normalized_weights.p'
save_file = data_path + fname
save_pickle_file(to_save, save_file)
# conn_data.to_csv(out_file) 


# In[12]:


np.sum(norm_conn_matrix, axis=0)


# In[13]:


np.sum(norm_conn_matrix, axis=1)


# In[ ]:




