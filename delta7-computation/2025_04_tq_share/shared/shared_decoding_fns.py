# April 4th 2023
# Editing to be compatible with the new Numpy random generation setup and to
# offer more options for ties in the get_winners function.
# January 25th 2023
# Shared functions for fly HD decoding

import numpy as np
import numpy.linalg as nla
# import numpy.random as nrd

def replace_empty_with_nan_list(inp_list):
    # Function to return a list of NaNs in place of an empty list, to help with
    # plotting distributions.
    
    # return([x if len(x) else np.array([np.nan, np.nan]) for x in inp_list])
    return([x if len(x) else np.array([float("nan"), float("nan")]) for x in inp_list])

def arg_max(x, ties='nan', rng=None):
    '''If x has a unique maximum, return the index of that. Otherwise,
    options are to break ties randomly or return NaN. Note that x
    should be a 1D array.'''

    max_idx = np.flatnonzero(x==np.max(x))
    # Unique max? 
    if len(max_idx)==1:
        return max_idx[0]
    elif ties == 'random':
        return rng.choice(max_idx)
    else:
        return np.nan

def get_winners(inp_list, k=4, n_trials=int(1e3), min_val=-1e5, norm=True, check_empty=False, 
    rng=None, break_ties='random'):
    '''Takes a list of lists. On each trial, draw k elements from each list and take the mean
    (or set to some low minimum value if list is empty) and figure out who had the max. If there is a 
    tie default is to return a random one.
    Did some testing and I think ok for now but could write some test code later.
    '''

    if check_empty and (np.min([len(x) for x in inp_list])==0):
        print('At least one direction is missing data so that direction will never be the max')
    
    n_options = len(inp_list)
    winners = np.zeros(n_options)

    for i in range(n_trials):
        # The "else min_val" takes care of the no data / empty list situation.
        selections = [np.mean(rng.choice(x, k)) if len(x) else min_val for x in inp_list]
        max_loc = arg_max(np.array(selections), ties=break_ties, rng=rng)
        winners[max_loc] = winners[max_loc] + 1
    if norm:
        winners = winners/np.sum(winners)
    return winners

def find_max_locations(A):
    '''For each row of A, set the location(s) of the max element to 1 and others to 0.
    This function helps to plot the max location and see how consistent it is.'''
    
    max_val_each_row = np.max(A, axis=1)
    max_val_array = np.tile(max_val_each_row, (A.shape[1], 1)).T
    max_loc_array = (A==max_val_array).astype(int)
    return max_loc_array

# Not sure if I actually implement this so commenting out
# def poisson_sample_and_count_winners(means, n_trials):
#     ''' Draw n_trials samples of Poisson distributed RVs with given means
#     and count winners. 
#     '''

#     # Maybe later think about seed with new random number generator setup
#     rng = nrd.default_rng()
#     n_options = len(means)
#     winners = np.zeros(n_options)
    
#     # Generate samples
#     samples = rng.poisson(lam=means, size=(n_trials, n_options))
    
#     for i in range(n_trials):
#         # Replace with my function once written
#         max_loc = np.argmax(samples[i])
#         winners[max_loc] = winners[max_loc] + 1
#     winners = winners/np.sum(winners)
#     return(winners)

# Given a list of means and a range of times, figure out when there's a unique winner
# with at least win_prob probability


