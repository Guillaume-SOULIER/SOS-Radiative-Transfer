import numpy as np
import matplotlib.pyplot as plt
#from tqdm import tqdm  # Removed for speed
import warnings
from SOS_Aer_global_va import MU_THRESHOLD, MU_EXTREME_THRESHOLD, MU_VERY_SMALL_THRESHOLD

# Suppress deprecation warnings for np.trapz
warnings.simplefilter('ignore', DeprecationWarning)

## Computation of radiance fields

# Computation of first order of scattering radiance
def I1_NumInt(tau, mu, tauStar, mu0, P0, alb, nb_angles):
    """
    Compute the first order of scattering radiance (I1) for all layers and angles.
    Downward: mu in [-1, 0), Upward: mu in [0, 1].
    Uses vectorized numpy operations for speed.
    """
    nb_layers = len(tau)
    I1 = np.zeros((nb_layers, 2*nb_angles))

    # Precompute exponentials for all tau and mu0 (used in both up and down)
    exp_tau_mu0 = np.exp(-tau / mu0)
    exp_tauStar_mu0 = np.exp(-tauStar / mu0)

    # --- Downward radiance (mu in [-1, 0)) ---
    mu_neg = mu[:nb_angles]
    P0_neg = P0[:nb_angles]
    for t in range(nb_layers):
        # Vectorized computation for all m except mu=0
        m_arr = np.arange(nb_angles-1)
        mu_m = mu_neg[m_arr]
        # General case for downward radiance
        sum_exp = exp_tau_mu0[t] - np.exp(+tau[t]/mu_m)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            I1[t, m_arr] = (mu0/(mu0+mu_m)) * (alb/(4*np.pi)) * P0_neg[m_arr] * sum_exp
        # Special case for mu=0
        I1[t, nb_angles-1] = (alb/(4*np.pi)) * (mu0/(mu0+mu_neg[nb_angles-1])) * P0_neg[nb_angles-1] * exp_tau_mu0[t]
        # Special case for |mu|=mu0
        mask_mu0 = np.abs(mu_m + mu0) < 0.0001
        if np.any(mask_mu0):
            I1[t, m_arr[mask_mu0]] = (alb/(4*np.pi)) * P0_neg[m_arr[mask_mu0]] * exp_tau_mu0[t] * tau[t] / mu0

    # --- Upward radiance (mu in [0, 1]) ---
    mu_pos = mu[nb_angles:]
    P0_pos = P0[nb_angles:]
    for t in range(nb_layers):
        # Special case for mu=0
        I1[t, nb_angles] = (alb/(4*np.pi)) * (mu0/(mu0+mu_pos[0])) * P0_pos[0] * exp_tau_mu0[t]
        # Vectorized for mu>0
        m_arr = np.arange(1, nb_angles)
        mu_m = mu_pos[m_arr]
        sum_exp = exp_tau_mu0[t] - exp_tauStar_mu0 * np.exp(-(tauStar-tau[t])/mu_m)
        I1[t, nb_angles + m_arr] = (mu0/(mu0+mu_m)) * (alb/(4*np.pi)) * P0_pos[m_arr] * sum_exp

    # Normalize values
    return I1*np.pi/mu0


# Computation of successive orders of scattering radiance
def Jn_NumInt(n, In_1, tau, mu, tauStar, mu0, P, alb, nb_angles): #In_1 : radiance scattered to the previous order I(n-1)
    """
    Compute the source function Jn for the nth order of scattering.
    Uses vectorized integration over mu for each layer.
    """
    nb_layers = len(tau)
    Jn = np.zeros((nb_layers, 2*nb_angles))
    # For each layer, integrate over all angles (vectorized)
    for t in range(nb_layers):
        # P[:,::-1] flips the phase function for integration over mu'
        # Multiply by In_1[t,:] (previous order radiance)
        Jn[t,:] = (alb/4) * np.trapz(P[:,::-1]*In_1[t,:], mu, axis=1)
    return Jn

# Computation of successive orders of scattering radiance
def In_NumInt(n, Jn, In_1, tau, mu, tauStar, mu0, P, alb, nb_angles, µ_1, µ_2): #In_1 : radaince scattered to the previous order I(n-1)
    """
    Compute the nth order of scattering radiance (In) for all layers and angles.
    Upward and downward directions are handled separately.
    Uses vectorized numpy operations for speed.
    """
    nb_layers = len(tau)
    In = np.zeros((nb_layers, 2*nb_angles))

    # --- Upward n-order of scattering radiance ---
    mu_pos = mu[nb_angles+1:2*nb_angles]
    for t in range(nb_layers):
        # Vectorized for mu>0 (excluding mu=0)
        m_arr = np.arange(nb_angles+1, 2*nb_angles)
        mu_m = mu[m_arr]
        tau_diff = tau[t:] - tau[t]  # shape: (nb_layers-t,)
        exp_term = np.exp(-tau_diff[:, None] / mu_m)  # shape: (nb_layers-t, nb_angles-1)
        # For very optically thick cases, use a different normalization
        if tauStar/mu_m[0] >= 50:
            In[t, m_arr] = np.trapz(Jn[t:, m_arr] * (exp_term / mu_m), tau[t:], axis=0)
        else:
            In[t, m_arr] = np.trapz(Jn[t:, m_arr] * exp_term, tau[t:], axis=0) / mu_m
        # Special case for mu=0
        In[t, nb_angles] = Jn[t, nb_angles]
        # Interpolation for μ → 0 region (unchanged)
        idx = nb_angles+1
        while np.abs((In[t,idx]-In[t,idx+1])-(In[t,idx+1]-In[t,idx+2]))>0.0001:
            idx+=1
        idx+=1
        for m in range(nb_angles+1,idx):
            weight = mu[m]/mu[idx]
            In[t,m]= (1-weight)*In[t, nb_angles] + weight*In[t,idx]

    # --- Downward n-order of scattering radiance ---
    mu_neg = mu[:nb_angles-1]
    for t in range(nb_layers):
        for m in range(nb_angles-1):
            if abs(mu[m]) < MU_THRESHOLD:
                # Use improved asymptotic expansion for μ → 0
                In[t,m] = improved_asymptotic_downward_radiance(Jn[:t+1,m], tau[:t+1], tau[t], mu[m])
            else:
                # Standard calculation for larger μ values
                tau_diff = tau[t] - tau[:t+1]
                exp_term = np.exp(tau_diff / mu[m])
                integrand = Jn[:t+1,m] * exp_term
                In[t,m] = -np.trapz(integrand, tau[:t+1]) / mu[m]
        # Improved interpolation for μ → 0 region (unchanged)
        if tauStar <= 0.0625: idx = int(0.005*nb_angles)
        elif tauStar <= 1: idx = int(0.02*nb_angles)
        elif tauStar < 4: idx = int(0.04*nb_angles)
        else: idx = int(0.06*nb_angles)
        for i in range(idx):
            In[t, nb_angles-1-i] = improved_limit_mu_down(In[t, :nb_angles], mu[:nb_angles], nb_angles, idx, i)
    return In


# Asymptotic expansion for downward radiance when μ → 0
def asymptotic_downward_radiance(Jn_slice, tau_slice, tau_t, mu):
    """
    Compute downward radiance using asymptotic expansion for μ → 0
    Based on the fact that exp(x/μ) → ∞ as μ → 0, but the integral should remain finite
    """
    if len(tau_slice) == 0:
        return 0.0
    
    # For very small μ, use the fact that the exponential term dominates
    # but we need to handle the integration carefully
    
    # Method 1: Use the fact that for μ → 0, the exponential term becomes a delta function
    # The integral becomes approximately Jn at the current layer
    if abs(mu) < 1e-6:
        return -Jn_slice[-1]  # Use the source function at current layer
    
    # Method 2: For small but not extremely small μ, use a more stable approach
    # Rewrite the integral to avoid the exponential explosion
    try:
        # Use log-space to avoid overflow
        log_integrand = np.log(np.abs(Jn_slice)) + (tau_t - tau_slice) / mu
        # Find the maximum to avoid overflow
        max_log = np.max(log_integrand)
        # Normalize and compute
        normalized_integrand = np.exp(log_integrand - max_log)
        integral = np.trapz(normalized_integrand, tau_slice) * np.exp(max_log)
        return -integral / mu
    except:
        # Fallback: use the source function at current layer
        return -Jn_slice[-1]


def doubling_method_downward_radiance(Jn_slice, tau_slice, tau_t, mu):
    """
    Alternative method using doubling approach for μ → 0
    This method is more stable for very small μ values
    """
    if len(tau_slice) == 0:
        return 0.0
    
    # For μ → 0, use the fact that the exponential term exp((tau_t - tau)/μ)
    # becomes very large for tau < tau_t, but the integral should remain finite
    
    # Use the fact that for μ → 0, the exponential term acts like a delta function
    # The main contribution comes from the layer at tau_t
    
    # Method: Use the source function at the current layer as the dominant term
    # and add a small correction from nearby layers
    
    if abs(mu) < 0.001:
        # For very small μ, use only the local source function
        return -Jn_slice[-1]
    
    # For moderately small μ, use a weighted average of nearby layers
    # The weight decreases exponentially with distance from current layer
    weights = np.exp((tau_t - tau_slice) / mu)
    # Normalize weights to avoid overflow
    max_weight = np.max(weights)
    normalized_weights = weights / max_weight
    
    # Compute weighted average
    weighted_integral = np.trapz(Jn_slice * normalized_weights, tau_slice)
    return -weighted_integral * max_weight / mu


def improved_asymptotic_downward_radiance(Jn_slice, tau_slice, tau_t, mu):
    """
    Improved asymptotic expansion with better numerical stability and physical accuracy for μ → 0.
    For very small μ, use Taylor expansion: I ≈ -Jn + μ dJn/dτ
    """
    if len(tau_slice) == 0:
        return 0.0
    
    # For μ → 0, use Taylor expansion for better accuracy
    if abs(mu) < MU_EXTREME_THRESHOLD:
        # Use Taylor expansion: I ≈ -Jn + μ dJn/dτ
        if len(tau_slice) > 1:
            dJ_dtau = (Jn_slice[-1] - Jn_slice[-2]) / (tau_slice[-1] - tau_slice[-2])
        else:
            dJ_dtau = 0.0
        return -Jn_slice[-1] + mu * dJ_dtau
    
    elif abs(mu) < MU_VERY_SMALL_THRESHOLD:
        # For very small μ, use the same Taylor expansion for smooth transition
        if len(tau_slice) > 1:
            dJ_dtau = (Jn_slice[-1] - Jn_slice[-2]) / (tau_slice[-1] - tau_slice[-2])
        else:
            dJ_dtau = 0.0
        return -Jn_slice[-1] + mu * dJ_dtau
    else:
        # For larger μ, use the previous approach (significant range)
        try:
            significant_range = 5 * abs(mu)
            significant_indices = np.where(tau_slice >= (tau_t - significant_range))[0]
            if len(significant_indices) == 0:
                return -Jn_slice[-1]
            tau_significant = tau_slice[significant_indices]
            Jn_significant = Jn_slice[significant_indices]
            integrand = Jn_significant * np.exp((tau_t - tau_significant) / mu)
            if np.any(np.isinf(integrand)) or np.any(np.isnan(integrand)):
                return -Jn_slice[-1]
            integral = np.trapz(integrand, tau_significant)
            return -integral / mu
        except (OverflowError, RuntimeWarning, ValueError):
            return -Jn_slice[-1]


# Improved interpolation for μ → 0 region
def improved_limit_mu_down(In_down, mu_down, nb_angles, idx, i):
    """
    Improved interpolation for the μ → 0 region using more stable methods
    """
    # Use more points for better interpolation
    n_points = min(5, idx)
    
    if n_points < 2:
        # Fallback to linear interpolation
        coeff_directeur = (In_down[-idx-2]-In_down[-idx-1])/(mu_down[-idx-2] - mu_down[-idx-1])
        ordo_orig = In_down[-idx-1]
        return coeff_directeur*(mu_down[-i-1] - mu_down[-idx-1]) + ordo_orig
    
    # Use polynomial interpolation with more points
    x_points = mu_down[-(idx+n_points):-idx]
    y_points = In_down[-(idx+n_points):-idx]
    
    # Fit a low-order polynomial to avoid oscillations
    if len(x_points) >= 3:
        # Ensure all are float64 to avoid linalg error
        x_fit = np.array(x_points, dtype=np.float64)
        y_fit = np.array(y_points, dtype=np.float64)
        mu_eval = float(mu_down[-i-1])
        coeffs = np.polyfit(x_fit, y_fit, min(2, len(x_fit)-1))
        return np.polyval(coeffs, mu_eval)
    else:
        # Linear interpolation
        coeff_directeur = (y_points[-1] - y_points[0]) / (x_points[-1] - x_points[0])
        return y_points[0] + coeff_directeur * (mu_down[-i-1] - x_points[0])


# Limit case for mu -> 0 : defining µ_1 and µ_2 for transition layer
def mu_approx_In(mu, nb_angles):
    idx = nb_angles
    while mu[idx]<0.009:
        idx += 1
    mu_1 = idx
    while mu[idx]<0.020:
        idx += 1
    mu_2 = idx
    return mu_1, mu_2

def limit_mu_down(In_down, mu_down, nb_angles, idx, i):
    coeff_directeur = (In_down[-idx-2]-In_down[-idx-1])/(mu_down[-idx-2] - mu_down[-idx-1])
    ordo_orig = In_down[-idx-1]
    return coeff_directeur*(mu_down[-i-1] - mu_down[-idx-1]) + ordo_orig

