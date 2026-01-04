#!/usr/bin/env python
# coding: utf-8

# June 19th, 2024  
# We've roughly converged on how to extract a basic profile from the connectivity data. This notebook generates the appropriate intermediate datasets.

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

curr_date=datetime.datetime.now().strftime('%Y_%m_%d')+'_'
#sd=int((time.time()%1)*(2**31))
# rng = nrd.default_rng(sd)
# print('Seed= ',sd)


# In[2]:


data_path = '../../data/2024_02_14_delta7_EPG_connectivity/delta7_to_delta7/'
# full_data_path = os.path.expanduser(data_path)
# file_list = glob.glob(data_path+'*.mat')
# fname1 = 'delta7_delta7_connection_subtype_combined.csv'
fname = 'delta7_delta7_connection.csv'
conn_data = pd.read_csv(data_path + fname)


# In[3]:


conn_data


# In[4]:


# Subtype list. Should be sorted by left glomerulus. Print to double check
subtype_list = sorted(conn_data['instance_pre'].unique())
# Check that same subtypes pre and post
print(sorted(conn_data['instance_post'].unique())==subtype_list)
print(subtype_list)
nsubtypes = len(subtype_list)

# List of all the neurons in the format (subtype, bodyId)
full_pre_neuron_list = conn_data[['instance_pre', 'bodyId_pre']].apply(tuple, axis=1)
full_post_neuron_list = conn_data[['instance_post', 'bodyId_post']].apply(tuple, axis=1)
pre_neuron_list = full_pre_neuron_list.unique()
post_neuron_list = full_post_neuron_list.unique()
# Check that same neurons pre and post
print(sorted(pre_neuron_list) == sorted(post_neuron_list))
neuron_list = sorted(pre_neuron_list)
# Make sure subtypes in neuron list match subtype list
print(np.all(pd.unique([x for x, _ in neuron_list])==subtype_list))


# In[5]:


# Few quick checks because bodyId is being read in as integer and want to make sure we don't overflow
print(sys.maxsize)
print(sys.maxsize- np.max(conn_data['bodyId_pre']))
print(9223372036854775807 - np.max(conn_data['bodyId_pre']))
print(2**63-1)


# In[6]:


# Check that the number of neurons in the form (subtype, bodyId) are just the same as the number of unique
# bodyIds, both pre and post
print(len(neuron_list))
print(len(conn_data['bodyId_pre'].unique()))
print(len(conn_data['bodyId_post'].unique()))


# In[7]:


# Store the total input weights into each neuron for later normalization
input_counts = {}
for i, nrn_info in enumerate(neuron_list):
    subtype, nrn = nrn_info
    # All inputs to this neuron
    input_conns = conn_data[conn_data['bodyId_post']==nrn]['weight']
    input_counts[nrn] = input_conns.sum()


# In[8]:


# Quick test that we did this right.
# Print tmp to see that entries look reasonable
tmp = conn_data[conn_data['bodyId_post']==5813061383]
print(np.sum(tmp['weight']))
print(input_counts[5813061383])


# In[9]:


conn_data


# In[10]:


# Now go through dataframe and add a new column of these normalized weights
conn_data['norm_weight'] = 0*conn_data['weight']
for i in conn_data.index:
    # Find the postsynaptic neuron for this connection
    post_nrn = conn_data.at[i, 'bodyId_post']
#     print(post_nrn)
    conn_data.at[i, 'norm_weight'] = conn_data.at[i, 'weight']/input_counts[post_nrn]


# In[11]:


# Few tests because normalized weights for first two seem the same!
print(input_counts[5813048042])
print(input_counts[941814787])
# conn_data[conn_data['bodyId_post']==5813048042]
print(conn_data[conn_data['bodyId_post']==5813048042]['weight'].sum())
len(conn_data[conn_data['bodyId_post']==5813048042])
len(conn_data[conn_data['bodyId_post']==941814787])


# In[12]:


conn_data


# In[13]:


print(input_counts[941482720])
print(conn_data.loc[[1823]])
print(18./583)


# Ok now we just want to save this new dataframe. 

# In[14]:


data_path = '../../results/2024_06_19/delta7_conn_with_normalized_weights/'
fname = 'delta7_delta7_connection_with_normalized_weights.csv'
out_file = data_path + fname
conn_data.to_csv(out_file) 


# Reload and run some checks to make sure it's the same thing
