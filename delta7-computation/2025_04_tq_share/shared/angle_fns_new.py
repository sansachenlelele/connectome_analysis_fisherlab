'''February 17th 2025
Putting together an updated set of functions for working with
angles (more generally periodic data).
'''

import numpy as np
from scipy.stats import linregress

def periodic_diff(x, y, period=2*np.pi):
    '''Circular difference between x and y, mapped to lie
    in (-period/2, period/2]
    Double check sign of difference
    '''
    
    if isinstance(x, float) and isinstance(y, float):
        diff = np.mod(x - y, period)
        if diff>period/2.:
            diff = diff - period
    else:
        diff = np.mod(np.array(x)-np.array(y), period)
        diff[diff>period/2.] = diff[diff>period/2.] - period
    return diff

def angular_diff(x, y):
    '''Circular difference between angles x and y in radians.
    Alternate way from periodic diff.'''

    angle_diffs = (x - y + np.pi) % (2 * np.pi) - np.pi
    return angle_diffs

def population_vector_decoder(activity, preferred_angles):
    Vx = np.sum(activity * np.cos(preferred_angles))
    Vy = np.sum(activity * np.sin(preferred_angles))
    
    theta_decoded = np.arctan2(Vy, Vx) % (2 * np.pi)
    
    return theta_decoded  # Output in radians

def compute_diffusion_curve(angles, max_tau=None):
    angles = np.unwrap(angles)  # Unwrap angles to avoid jumps at ±pi
    T = len(angles)
    
    if max_tau is None:
        max_tau = T // 2  # Limit max tau to half the data length
    
    taus = np.arange(1, max_tau)  # Time intervals to analyze
    variances = []

    for tau in taus:
        # Angle diffs, wrapped to [-pi, pi]
        angle_diffs = angular_diff(angles[tau:], angles[:-tau])
        variances.append(np.var(angle_diffs))  # Compute variance

    return taus, np.array(variances)

def compute_diffusion_constant(taus, variances, fit_range=None):
    """
    Compute the diffusion constant D by fitting a linear model to Var(Δθ) vs. τ.
    
    Parameters:
    - taus: Array of time intervals (τ).
    - variances: Array of variances of angle changes at each τ.
    - fit_range: Tuple (τ_min, τ_max) specifying the range to fit. If None, fits the first half of the data.
    
    Returns:
    - D: Estimated diffusion constant.
    - fit_slope: Slope of the linear fit.
    - fit_intercept: Intercept of the linear fit.
    """

    if fit_range is None:
        fit_range = (taus[0], taus[len(taus) // 2])  # Fit first half by default
    
    # Select data in the chosen range
    mask = (taus >= fit_range[0]) & (taus <= fit_range[1])
    taus_fit = taus[mask]
    variances_fit = variances[mask]

    # Perform linear regression
    slope, intercept, _, _, _ = linregress(taus_fit, variances_fit)

    # Compute diffusion constant
    D = slope / 2
    return D, slope, intercept


    