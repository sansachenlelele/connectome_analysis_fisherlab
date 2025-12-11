'''April 24th 2024
Some helper functions for the connectivity data analysis and simulation.
'''

import numpy as np

def norm01(x):
    '''Rescale the vector or matrix x to lie between 0 and 1.'''
    return (x - np.min(x))/(np.max(x) - np.min(x))

def mat_normalize(A, axis=0):
    '''Return a normalized version of A so that the sum is 1
    across axis. Currently only works for 2D matrices / axes 0 and 1.'''

    if axis==0:
        col_sums = np.sum(A, axis=0)
        return A/col_sums
    elif axis==1:
        row_sums = np.sum(A, axis=1)
        return (A.T/row_sums).T

def thresh_matrix(A, threshold):
    B = np.array(A)
    B[B<=threshold]=0.
    B[B>threshold]=1.
    return(B)


def diag_align(A, offset=0):
    '''Align the rows of A so that the diagonals all line up. The main diagonal
    ends up at position offset.'''

    B = np.zeros_like(A)
    for i in range(len(B)):
        B[i] = np.roll(A[i], -i+offset)

    return(B)

def gather_over_subtype(W, subtype_list, subtype_idx):
    '''
    Take a neuron x neuron connectivity matrix W, a list of neuron subtypes, and a dictionary that returns 
    the neuron indices corresponding to each subtype. Computes and returns a subtype x subtype connectivity 
    matrix, where S[i,j] is some aggregate connection strength between subtype j and subtype i 
    (from vs. to determined by ordering chosen for W). For now, we're summing over columns and averaging over rows.
    This assumes that columns are presynaptic and rows are postsynaptic. So we sum to compute the total input each
    neuron of subtype i gets from all the neurons in subtype j. We then average to figure out the input a typical
    neuron of subtype i sees.
    
    '''
    S = np.zeros((len(subtype_list), len(subtype_list)))
    
    for i, row_subt in enumerate(subtype_list):
        for j, col_subt in enumerate(subtype_list):
            row_idx = subtype_idx[row_subt]
            col_idx = subtype_idx[col_subt]
        
            conns_i_j = W[np.ix_(row_idx, col_idx)]
            S[i,j] = np.mean(np.sum(conns_i_j, axis=1))
    
    return S    

# Later do via broadcasting
# conn_mat_row_norm = np.array(conn_mat)
# conn_mat_col_norm = np.array(conn_mat)

# for i in range(len(conn_mat)):
#     conn_mat_row_norm[i] = conn_mat_row_norm[i]/np.sum(conn_mat[i])
#     conn_mat_col_norm[:,i] = conn_mat_col_norm[:,i]/np.sum(conn_mat[:,i])

# print(np.sum(conn_mat_row_norm, axis=1))
# print(np.sum(conn_mat_col_norm, axis=0))

