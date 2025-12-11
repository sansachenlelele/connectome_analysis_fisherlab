# March 2nd 2023
# File functions

import pickle

def save_pickle_file(data, filename):
    # Store data (serialize)
    with open(filename, 'wb') as handle:
        #pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(data, handle)

def load_pickle_file(filename):
    # Load data (deserialize)
    with open(filename, 'rb') as handle:
        data = pickle.load(handle)
    return(data)



# Old versions we used to use
# def load_pickle_file(filename):
#     fr = open(filename, 'rb')
#     data = pickle.load(fr)
#     fr.close()
#     return data


# def save_pickle_file(data, filename):
#     fw = open(filename, 'wb')
#     pickle.dump(data, fw)
#     fw.close()
#     return 1
