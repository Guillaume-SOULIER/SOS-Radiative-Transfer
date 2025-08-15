import numpy as np
from SOS_Aer_global_va import MU_EXTREME_THRESHOLD, MU_THRESHOLD, MU_VERY_SMALL_THRESHOLD

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

