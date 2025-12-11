# July 5th 2022
# Copied over ring network functions from HD project. Simplifying, and pushing
# other stuff below and commenting it out for later reference.
# Also changed simulation functions to have time / observations on rows

import numpy as np

# Functions to set up recurrent connectivity
def bf_conn_profile(theta, A=1, k1=1, k2=0.3):
	'''Recurrent connection strength for neurons separated by angle theta, 
	derived from Burak & Fiete 2012'''
	tmp = np.cos(theta)-1
	return A*np.exp(k1*tmp) - A*np.exp(k2*tmp)

def get_W_and_pref_angles(n_neurons, conn_profile):
	'''Turn a connectivity profile function into a connection matrix. 
	'''

	theta_n = 2*np.pi*np.arange(n_neurons)/n_neurons
	W = np.zeros((n_neurons, n_neurons))

	for i in range(n_neurons):
		W[i] = conn_profile(theta_n[i] - theta_n)

	return theta_n, W

# f-I curves. 
def exp_fi(g):
	'''Exponential f-I curve from Burak & Fiete'''
	return 10*np.exp(g)/(1e-2)

def tanh_fi(g):
	'''Tanh (sigmoid) f-I curve from Burak & Fiete'''
	return 0.2*(1+np.tanh(g+4))/(1e-2)


# Steps to simulate network
# Given s(t), compute the new input g = Ws + b and the new output rate
# r_i = phi(g_i)

def simulate_lnp_net(W, params):
	'''Makes the assumption that dt is small enough that Poisson spiking
	can be replaced by Bernoulli process with p = rate*dt.
	params has nSteps, dt, tau, bias, fi and possibly ics.'''

	n_neurons = W.shape[0]
	s = np.zeros((params['nSteps'], n_neurons))
	rate_mat = np.zeros_like(s)
	if 'ics' in params:
		s[0] = params['ics']

	s_decay = np.exp(-params['dt']/params['tau'])

	for i in range(params['nSteps']-1):
		g = np.dot(W, s[i]) + params['bias']
		rates = params['fi'](g)
		# Assume that dt is small enough that Bernoulli spiking
		spikes = (np.random.rand(n_neurons)<(rates*params['dt'])).astype(int)
		s[i+1] = s_decay*s[i] + spikes

		# Store to test
		# Changed to i rather than i+1 because these are rates calculated from s[i]
		rate_mat[i] = rates

	return s, rate_mat

def simulate_lnp_net_with_vel_flucts(W, W_left, W_right, params):
	'''Now assume that the contribution of W_left and W_right undergoes fluctuations.
	Makes the assumption that dt is small enough that Poisson spiking
	can be replaced by Bernoulli process with p = rate*dt.
	params has nSteps, dt, tau, bias, fi and possibly ics.'''
	n_neurons = W.shape[0]
	s = np.zeros((n_neurons, params['nSteps']))
	rate_mat = np.zeros_like(s)
	if 'ics' in params:
		s[:,0] = params['ics']
	s_decay = np.exp(-params['dt']/params['tau'])

	# Set up alpha
	alpha = np.zeros(params['nSteps'])
	alpha_det = 1+(-1./params['tau'])*params['dt']
	alpha_noise_sigma=params['alpha_sigma']*np.sqrt(params['dt']*2/params['tau'])
	
	W_comb = W + 0.1*W_left + 0.1*W_right

	for i in range(params['nSteps']-1):
		alpha[i+1] = alpha_det*alpha[i] + alpha_noise_sigma*np.random.randn()
		W_curr = W_comb + alpha[i]*W_left - alpha[i]*W_right
		g = np.dot(W_curr, s[:,i]) + params['bias']
		rates = params['fi'](g)
		# Assume that dt is small enough that Bernoulli spiking
		spikes = (np.random.rand(n_neurons)<(rates*params['dt'])).astype(int)
		s[:,i+1] = s_decay*s[:,i] + spikes

		# Store to test. Changing to i rather i+1 because rates
		# are calculated from s[:,i]
		rate_mat[:,i] = rates

	return s, rate_mat, alpha

def simulate_lnp_net_save_at_interval(W, params):
	'''Makes the assumption that dt is small enough that Poisson spiking
	can be replaced by Bernoulli process with p = rate*dt.
	params has nSteps, dt, tau, bias, fi and possibly ics.'''
	n_neurons = W.shape[0]
	nSamples = int(np.round(params['nSteps']/params['save_step']))

	s_mat = np.zeros((n_neurons, nSamples))
	rate_mat = np.zeros_like(s_mat)
	
	if 'ics' in params:
		curr_s = params['ics']
	else:
		curr_s = np.zeros(n_neurons)

	s_decay = np.exp(-params['dt']/params['tau'])

	for i in range(params['nSteps']-1):
		g = np.dot(W, curr_s) + params['bias']
		rates = params['fi'](g)
		
		if i%params['save_step']==0:
			# print i
			j = int(np.round(i/params['save_step']))
			s_mat[:,j] = curr_s
			rate_mat[:,j] = rates

		# Assume that dt is small enough that Bernoulli spiking
		spikes = (np.random.rand(n_neurons)<(rates*params['dt'])).astype(int)
		curr_s = s_decay*curr_s + spikes

	return s_mat, rate_mat

def simulate_lnp_net_with_vel_flucts_save_at_interval(W, W_left, W_right, params):
	'''Now assume that the contribution of W_left and W_right undergoes fluctuations.
	Makes the assumption that dt is small enough that Poisson spiking
	can be replaced by Bernoulli process with p = rate*dt.
	params has nSteps, dt, tau, bias, fi and possibly ics. Also save_step'''
	n_neurons = W.shape[0]
	nSamples = int(np.round(params['nSteps']/params['save_step']))

	s_mat = np.zeros((n_neurons, nSamples))
	rate_mat = np.zeros_like(s_mat)

	if 'ics' in params:
		curr_s = params['ics']
	else:
		curr_s = np.zeros(n_neurons)

	s_decay = np.exp(-params['dt']/params['tau'])

	# Set up alpha
	alpha_mat = np.zeros(nSamples)
	curr_alpha = 0
	alpha_det = 1+(-1./params['tau'])*params['dt']
	alpha_noise_sigma=params['alpha_sigma']*np.sqrt(params['dt']*2/params['tau'])
	
	W_comb = W + 0.1*W_left + 0.1*W_right

	for i in range(params['nSteps']-1):
		# First, get everything up to current time step and save it if appropriate
		W_curr = W_comb + curr_alpha*W_left - curr_alpha*W_right
		g = np.dot(W_curr, curr_s) + params['bias']
		rates = params['fi'](g)

		if i%params['save_step']==0:
			j = int(np.round(i/params['save_step']))
			s_mat[:,j] = curr_s
			rate_mat[:,j] = rates
			alpha_mat[j] = curr_alpha

		# Now update whatever needs to be updated
		curr_alpha = alpha_det*curr_alpha + alpha_noise_sigma*np.random.randn()
		
		# Assume that dt is small enough that Bernoulli spiking
		spikes = (np.random.rand(n_neurons)<(rates*params['dt'])).astype(int)
		curr_s = s_decay*curr_s + spikes

	return s_mat, rate_mat, alpha_mat



def simulate_lnp_net_bg_input(W, params):
	'''Makes the assumption that dt is small enough that Poisson spiking
	can be replaced by Bernoulli process with p = rate*dt.
	params has nSteps, dt, tau, bias, fi and possibly ics.'''
	n_neurons = W.shape[0]
	s = np.zeros((n_neurons, params['nSteps']))
	rate_mat = np.zeros_like(s)
	if 'ics' in params:
		s[:,0] = params['ics']

	s_decay = np.exp(-params['dt']/params['tau'])

	for i in range(params['nSteps']-1):
		g = np.dot(W, s[:,i]) + params['bias'] + params['bg_input'][i]
		rates = params['fi'](g)
		# Assume that dt is small enough that Bernoulli spiking
		spikes = (np.random.rand(n_neurons)<(rates*params['dt'])).astype(int)
		s[:,i+1] = s_decay*s[:,i] + spikes

		# Store to test
		rate_mat[:,i+1] = rates

	return s, rate_mat

def simulate_lnp_net_save_at_interval_bg_input(W, params):
	'''Makes the assumption that dt is small enough that Poisson spiking
	can be replaced by Bernoulli process with p = rate*dt.
	params has nSteps, dt, tau, bias, fi and possibly ics.'''
	n_neurons = W.shape[0]
	nSamples = int(params['nSteps']/params['save_step'])

	s_mat = np.zeros((n_neurons, nSamples))
	rate_mat = np.zeros_like(s_mat)
	
	if 'ics' in params:
		curr_s = params['ics']
	else:
		curr_s = np.zeros(n_neurons)

	s_decay = np.exp(-params['dt']/params['tau'])

	for i in range(params['nSteps']-1):
		g = np.dot(W, curr_s) + params['bias'] + params['bg_input'][i]
		rates = params['fi'](g)
		
		if i%params['save_step']==0:
			# print i
			j = int(np.round(i/params['save_step']))
			s_mat[:,j] = curr_s
			rate_mat[:,j] = rates

		# Assume that dt is small enough that Bernoulli spiking
		spikes = (np.random.rand(n_neurons)<(rates*params['dt'])).astype(int)
		curr_s = s_decay*curr_s + spikes

	return s_mat, rate_mat

