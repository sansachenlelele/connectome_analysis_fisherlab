#!/usr/bin/env python
# coding: utf-8

# June 27th 2024  
# Use the weights to calculate the average connectivity profile in various ways and save them. This code tests manually too.

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
import conn_data_analysis_fns as cdaf

curr_date=datetime.datetime.now().strftime('%Y_%m_%d')+'_'
#sd=int((time.time()%1)*(2**31))
# rng = nrd.default_rng(sd)
# print('Seed= ',sd)


# In[2]:


data_path = '../../results/2024_06_26/delta7_rec_matrices/'
fname = 'delta7_delta7_connections_rec_matrices_with_normalized_weights.p'
conn_matrices = load_pickle_file(data_path + fname)


# In[3]:


neuron_list = conn_matrices['neuron_list']
subtype_list = conn_matrices['subtype_list']
subtype_idx = conn_matrices['subtype_idx']
raw_conn_matrix = conn_matrices['raw_conn_matrix']
norm_conn_matrix = conn_matrices['norm_conn_matrix']


# In[4]:


S_fn = cdaf.gather_over_subtype(norm_conn_matrix.T, subtype_list, subtype_idx)


# In[5]:


# Compute manually.
# Note that this version has pre on rows, so is transpose
W = norm_conn_matrix.T
S_manual = np.zeros((len(subtype_list), len(subtype_list)))
for i, source_subt in enumerate(subtype_list):
    source_sub_idx = subtype_idx[source_subt]
#     print(i, source_subt, source_sub_idx)
    
    # Fictive activity vector in which neurons of this subtype
    # are active and everyone is silent
    x = np.zeros(len(neuron_list))
    x[source_sub_idx] = 1.
    
    # What inhibition does everyone else feel if this were the
    # network state?
    y = W @ x

    # Now we want to average this by subtype to ask what average
    # inhibition is felt by neurons of each subtype
    for j, target_subt in enumerate(subtype_list):
        target_sub_idx = subtype_idx[target_subt]
        S_manual[i,j] = np.mean(y[target_sub_idx])

print(np.allclose(S_manual, S_fn.T))


# In[6]:


shifted_S = cdaf.diag_align(S_fn, offset=5)
avg_profile_fn = np.mean(shifted_S, axis=0)


# In[7]:


avg_profile_manual = np.zeros_like(S_fn[0])
for i in range(len(S_fn)):
    vc1 = np.roll(S_fn[i], -i+5)
#     print(vc1 - shifted_S[i])
    avg_profile_manual = avg_profile_manual + vc1

avg_profile_manual = avg_profile_manual/len(S_fn)

print(np.allclose(avg_profile_fn, avg_profile_manual))


# In[8]:


fig, ax = plt.subplots(1,2,figsize=(12,6))
for i in range(len(shifted_S)):
    ax[0].plot(np.arange(len(shifted_S[i])), shifted_S[i])
    ax[1].plot(np.arange(len(shifted_S[i])), shifted_S[i], alpha=0.5)

ax[1].plot(np.arange(len(shifted_S[i])), avg_profile_fn, color='k', marker='.')


# In[12]:


left_glom_idx = defaultdict(list)
right_glom_idx = defaultdict(list)

for subtype in subtype_list:
    rel_nrn_idx = subtype_idx[subtype]
    x = subtype[13:] # Get rid of first part of string
    l_glom = x.partition('R')[0]
    r_glom = ('R' + x.partition('R')[2])[:-2]
    print(subtype, l_glom, r_glom)
    
    left_glom_idx[l_glom].extend(rel_nrn_idx)
    right_glom_idx[r_glom].extend(rel_nrn_idx)

left_glom_list = sorted(left_glom_idx.keys())
right_glom_list = sorted(right_glom_idx.keys())


# In[13]:


left_glom_list


# In[14]:


right_glom_list


# In[27]:


print([(k[13:-2], len(v)) for k, v in subtype_idx.items()])
print('\n')
print([(k, len(v)) for k, v in left_glom_idx.items()])
print([(k, len(v)) for k, v in right_glom_idx.items()])


# In[16]:


right_glom_idx


# In[17]:


S_left = cdaf.gather_over_subtype(norm_conn_matrix.T, left_glom_list, left_glom_idx)
shifted_S_left = cdaf.diag_align(S_left, offset=4)
avg_profile_left = np.mean(shifted_S_left, axis=0)

fig, ax = plt.subplots(1,2,figsize=(12,6))
for i in range(len(shifted_S_left)):
    ax[0].plot(np.arange(len(shifted_S_left[i])), shifted_S_left[i])
    ax[1].plot(np.arange(len(shifted_S_left[i])), shifted_S_left[i], alpha=0.5)

ax[1].plot(np.arange(len(shifted_S_left[i])), avg_profile_left, color='k', marker='.')


# In[18]:


S_right = cdaf.gather_over_subtype(norm_conn_matrix.T, right_glom_list, right_glom_idx)
shifted_S_right = cdaf.diag_align(S_right, offset=4)
avg_profile_right = np.mean(shifted_S_right, axis=0)

fig, ax = plt.subplots(1,2,figsize=(12,6))
for i in range(len(shifted_S_right)):
    ax[0].plot(np.arange(len(shifted_S_right[i])), shifted_S_right[i])
    ax[1].plot(np.arange(len(shifted_S_right[i])), shifted_S_right[i], alpha=0.5)

ax[1].plot(np.arange(len(shifted_S_right[i])), avg_profile_right, color='k', marker='.')


# In[19]:


fig, ax = plt.subplots(1,2,figsize=(12,6))
for i in range(len(shifted_S_right)):
    ax[0].plot(np.arange(len(shifted_S_right[i])), shifted_S_left[i] + shifted_S_right[i])
    ax[1].plot(np.arange(len(shifted_S_right[i])), shifted_S_left[i] + shifted_S_right[i], alpha=0.5)

ax[1].plot(np.arange(len(shifted_S_right[i])), avg_profile_left + avg_profile_right, color='k', marker='.')


# In[ ]:




