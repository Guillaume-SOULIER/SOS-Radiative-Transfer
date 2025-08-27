import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from SOS_Aer_global_va import load_atm_parameters, load_atm_phase_functions, load_aer_parameters, load_aer_phase_functions, save_atm_phase_functions, save_aer_phase_functions, save_atm_parameters, save_aer_parameters
from SOS_Aer_fwc_data import phase_func_FWC, mu_fwc
import miepython
import os
from multiprocessing import Pool, cpu_count

## Computation of phase function

def phase_func(mol, phase_fun, g, r, lambda0, indx, nb_angles, mu, mu0, N0, r_m, sig):

    print(f"Computing {mol} phase function:")


    if mol == 'atm': 
        last_mu0, last_g, last_phase_func = load_atm_parameters()
        P0, P = load_atm_phase_functions()
    elif mol =='aer': 
        last_mu0, last_g, last_phase_func = load_aer_parameters()
        P0, P = load_aer_phase_functions()

    if (
        P0 is None or P is None or
        P0.shape[0] != 2*nb_angles or
        len(P.shape) != 2 or
        P.shape[0] != 2*nb_angles or P.shape[1] != 2*nb_angles or
        last_mu0 is None or last_g is None or last_phase_func is None or
        (last_mu0 is not None and not np.isclose(mu0, last_mu0)) or
        (last_g is not None and not np.isclose(g, last_g)) or
        (last_phase_func is not None and last_phase_func != phase_fun)
    ):
        print(f" - Memory is empty or incompatible, computing {phase_fun} phase function")
        
        if phase_fun == 'iso':
            P0, P = isotropic(nb_angles, mu)
        elif phase_fun == 'hg': 
            P0, P = henyey_greenstein(nb_angles, mu, mu0, g)
        elif phase_fun == 'mie':
            P0, P = mie(nb_angles, mu, mu0, indx, r, lambda0)
        elif phase_fun == 'fwc':
            P0, P = fwc(nb_angles, mu, mu0)
        elif phase_fun == 'rayleigh':
            P0, P = rayleigh(nb_angles, mu, mu0)
        elif phase_fun == 'eva' or 'wildfire':
            P0, P = log_normal_mie(nb_angles, mu, mu0, lambda0, indx, N0, r_m, sig, phase_fun)
        
        if mol == 'atm':
            save_atm_phase_functions(P, P0)
            save_atm_parameters(mu0, g, phase_fun)
        elif mol == 'aer':
            save_aer_phase_functions(P, P0)
            save_aer_parameters(mu0, g, phase_fun)
    
    else:
        print(' - Memory is not empty, using last phase function')
    
    
    if P0 is None or P is None: raise ValueError(f"Phase function for {mol} is None! Check memory or computation.")
    

    return P0, P




def isotropic(nb_angles, mu):
    
    # Phase function for 1st order of scattering : P(µ, µ0)
    P0 = np.ones(2*nb_angles)

    # Phase function for n-th order of scattering : P(µ, µ')
    P = 2*np.ones((2*nb_angles, 2*nb_angles)) # normalization to 2
    
    return P0, P


def rayleigh(nb_angles, mu, mu0):

    nb_phi = 25
    phi = np.linspace(0, np.pi, nb_phi)
    phi0=0

    # Phase function for 1st order of scattering : P(µ, µ0)

    P_pos = np.zeros((2*nb_angles, nb_phi))
    P_neg = np.zeros((2*nb_angles, nb_phi))
    P0 = np.zeros(2*nb_angles)

    for m in tqdm(range(2*nb_angles), desc="P(µ, µ0)", leave=True):

        for p in range(nb_phi): # loop on azimuth Phi

            mu_diff_pos = - (mu[m]*mu0 + np.sqrt(1-mu0*mu0)*np.sqrt(1-mu[m]*mu[m])*np.cos(phi0-phi[p])) # theta_diff = pi - (theta+theta0) # incidence angle : µ0
            P_pos[m, p] = (3/4)*(1+mu_diff_pos*mu_diff_pos)

            mu_diff_neg = - (mu[m]*mu0 - np.sqrt(1-mu0*mu0)*np.sqrt(1-mu[m]*mu[m])*np.cos(phi0-phi[p])) # incidence angle : (µ0, phi0 + pi)
            P_neg[m, p] = (3/4)*(1+mu_diff_neg*mu_diff_neg)

        P0[m] = np.trapz(P_pos[m, :]+P_neg[m,:], phi)/(4*np.pi) # average on azimuth angle Phi

    P0 = P0/np.trapz(P0, mu) *2 # normalization

    # Phase function for n-th order of scattering : P(µ, µ')

    P = np.zeros((2*nb_angles, 2*nb_angles)) # only defined for µ>0

    P_pos = np.zeros(nb_phi)
    P_neg = np.zeros(nb_phi)

    for n in tqdm(range(2*nb_angles), desc="P(µ, µ')", leave=True): # Loop on -1<µ'<1 (incidence angle as in P(µ, µ0))

        for m in range(n, 2*nb_angles): # Loop on 0<µ<1 (exit angle) # P(µ,µ') = P(µ',µ)

            for p in range(nb_phi):

                cosm_cosn = mu[m]*mu[n]
                sinm_sinn = np.sqrt(1-mu[n]*mu[n])*np.sqrt(1-mu[m]*mu[m])


                mu_diff_pos = - (cosm_cosn + sinm_sinn*np.cos(phi0-phi[p])) # theta_diff = pi - (theta-theta0) # incidence angle : µ0 # minus sign comes from theta_diff = pi - ()
                P_pos[p] = (3/4)*(1+mu_diff_pos*mu_diff_pos)

                mu_diff_neg = - (cosm_cosn - sinm_sinn*np.cos(phi0-phi[p])) # incidence angle : (µ0, phi0 + pi) -> completes the other part of ring (phi 0 -> pi & phi+pi pi -> 2pi)
                P_neg[p] = (3/4)*(1+mu_diff_neg*mu_diff_neg)

            P[m,n] = np.trapz(P_pos+P_neg, phi)/(2*np.pi)
            P[n,m] = P[m,n] # symmetry

        P[:,n] = 4 * P[:,n]/(np.trapz(P[:,n], mu[:])) # normalization (for a given entrance angle, sum of exit probabilities equal to 2)
    
    return P0, P

    # incidence angle on a circle : location of the sun is graphed by phi moving 0-2pi
    # theta : 0 to pi
    # phi : à to 2*pi



def henyey_greenstein(nb_angles, mu,  mu0, g): 
    
    nb_phi = 25
    phi = np.linspace(0, np.pi, nb_phi)
    phi0=0

    # Phase function for 1st order of scattering : P(µ, µ0)

    P_pos = np.zeros((2*nb_angles, nb_phi))
    P_neg = np.zeros((2*nb_angles, nb_phi))
    P0 = np.zeros(2*nb_angles)

    for m in tqdm(range(2*nb_angles), desc="P(µ, µ0)", leave=True):

        for p in range(nb_phi): # loop on azimuth Phi

            mu_diff_pos = - (mu[m]*mu0 + np.sqrt(1-mu0*mu0)*np.sqrt(1-mu[m]*mu[m])*np.cos(phi0-phi[p])) # theta_diff = pi - (theta+theta0) # incidence angle : µ0
            P_pos[m, p] = (1-g*g) / ((1 + g*g - 2*g*mu_diff_pos)**(1.5))

            mu_diff_neg = - (mu[m]*mu0 - np.sqrt(1-mu0*mu0)*np.sqrt(1-mu[m]*mu[m])*np.cos(phi0-phi[p])) # incidence angle : (µ0, phi0 + pi)
            P_neg[m, p] = (1-g*g) / ((1 + g*g - 2*g*mu_diff_neg)**(1.5))

        P0[m] = np.trapz(P_pos[m, :]+P_neg[m,:], phi)/(4*np.pi) # average on azimuth angle Phi

    P0 = P0/np.trapz(P0, mu) *2 # normalization

    # Phase function for n-th order of scattering : P(µ, µ')

    P = np.zeros((2*nb_angles, 2*nb_angles)) # only defined for µ>0

    P_pos = np.zeros(nb_phi)
    P_neg = np.zeros(nb_phi)

    for n in tqdm(range(2*nb_angles), desc="P(µ, µ')", leave=True): # Loop on -1<µ'<1 (incidence angle as in P(µ, µ0))

        for m in range(n, 2*nb_angles): # Loop on 0<µ<1 (exit angle) # P(µ,µ') = P(µ',µ)

            for p in range(nb_phi):

                cosm_cosn = mu[m]*mu[n]
                sinm_sinn = np.sqrt(1-mu[n]*mu[n])*np.sqrt(1-mu[m]*mu[m])


                mu_diff_pos = - (cosm_cosn + sinm_sinn*np.cos(phi0-phi[p])) # theta_diff = pi - (theta-theta0) # incidence angle : µ0 # minus sign comes from theta_diff = pi - ()
                P_pos[p] = (1-g*g) / ((1 + g*g - 2*g*mu_diff_pos)**(1.5))

                mu_diff_neg = - (cosm_cosn - sinm_sinn*np.cos(phi0-phi[p])) # incidence angle : (µ0, phi0 + pi) -> completes the other part of ring (phi 0 -> pi & phi+pi pi -> 2pi)
                P_neg[p] = (1-g*g) / ((1 + g*g - 2*g*mu_diff_neg)**(1.5))

            P[m,n] = np.trapz(P_pos+P_neg, phi)/(2*np.pi)
            P[n,m] = P[m,n] # symmetry

        P[:,n] = 4 * P[:,n]/(np.trapz(P[:,n], mu[:])) # normalization (for a given entrance angle, sum of exit probabilities equal to 2)
    
    return P0, P

    # incidence angle on a circle : location of the sun is graphed by phi moving 0-2pi
    # theta : 0 to pi
    # phi : à to 2*pi


def interpolate_fwc_phase(mu_diff):
    """
    Linear interpolation of the FWC phase function for a given scattering angle.
    
    Args:
        mu_diff: scattering angle (cosine of the scattering angle)
        
    Returns:
        Interpolated value of the FWC phase function
    """
    # Ensure mu_diff is within the range [-1, 1]
    mu_diff = np.clip(mu_diff, -1, 1)
    
    # Find the indices of the two closest points
    idx = np.searchsorted(mu_fwc, mu_diff)
    
    # Handle boundary cases
    if idx == 0:
        return phase_func_FWC[0]
    elif idx >= len(mu_fwc):
        return phase_func_FWC[-1]
    
    # Linear interpolation
    mu_low = mu_fwc[idx-1]
    mu_high = mu_fwc[idx]
    phase_low = phase_func_FWC[idx-1]
    phase_high = phase_func_FWC[idx]
    
    # Calculate interpolation weight
    weight = (mu_diff - mu_low) / (mu_high - mu_low)
    
    # Linear interpolation
    phase_interpolated = phase_low + weight * (phase_high - phase_low)
    
    return phase_interpolated

def fwc(nb_angles, mu, mu0):
    
    nb_phi = 25
    phi = np.linspace(0, np.pi, nb_phi)
    phi0=0

    # Phase function for 1st order of scattering : P(µ, µ0)

    P_pos = np.zeros((2*nb_angles, nb_phi))
    P_neg = np.zeros((2*nb_angles, nb_phi))
    P0 = np.zeros(2*nb_angles)

    for m in tqdm(range(2*nb_angles), desc="P(µ, µ0)", leave=True):

        for p in range(nb_phi): # loop on azimuth Phi

            mu_diff_pos = - (mu[m]*mu0 + np.sqrt(1-mu0*mu0)*np.sqrt(1-mu[m]*mu[m])*np.cos(phi0-phi[p])) # theta_diff = pi - (theta+theta0) # incidence angle : µ0
            P_pos[m, p] = interpolate_fwc_phase(mu_diff_pos)

            mu_diff_neg = - (mu[m]*mu0 - np.sqrt(1-mu0*mu0)*np.sqrt(1-mu[m]*mu[m])*np.cos(phi0-phi[p])) # incidence angle : (µ0, phi0 + pi)
            P_neg[m, p] = interpolate_fwc_phase(mu_diff_neg)

        P0[m] = np.trapz(P_pos[m, :]+P_neg[m,:], phi)/(4*np.pi) # average on azimuth angle Phi

    P0 = P0/np.trapz(P0, mu) *2 # normalization

    # Phase function for n-th order of scattering : P(µ, µ')

    P = np.zeros((2*nb_angles, 2*nb_angles)) # only defined for µ>0

    P_pos = np.zeros(nb_phi)
    P_neg = np.zeros(nb_phi)

    for n in tqdm(range(2*nb_angles), desc="P(µ, µ')", leave=True): # Loop on -1<µ'<1 (incidence angle as in P(µ, µ0))

        for m in range(n, 2*nb_angles): # Loop on 0<µ<1 (exit angle) # P(µ,µ') = P(µ',µ)

            for p in range(nb_phi):

                cosm_cosn = mu[m]*mu[n]
                sinm_sinn = np.sqrt(1-mu[n]*mu[n])*np.sqrt(1-mu[m]*mu[m])


                mu_diff_pos = - (cosm_cosn + sinm_sinn*np.cos(phi0-phi[p])) # theta_diff = pi - (theta-theta0) # incidence angle : µ0 # minus sign comes from theta_diff = pi - ()
                P_pos[p] = interpolate_fwc_phase(mu_diff_pos)

                mu_diff_neg = - (cosm_cosn - sinm_sinn*np.cos(phi0-phi[p])) # incidence angle : (µ0, phi0 + pi) -> completes the other part of ring (phi 0 -> pi & phi+pi pi -> 2pi)
                P_neg[p] = interpolate_fwc_phase(mu_diff_neg)

            P[m,n] = np.trapz(P_pos+P_neg, phi)/(2*np.pi)
            P[n,m] = P[m,n] # symmetry

        P[:,n] = 4 * P[:,n]/(np.trapz(P[:,n], mu[:])) # normalization (for a given entrance angle, sum of exit probabilities equal to 2)
    
    return P0, P

    # incidence angle on a circle : location of the sun is graphed by phi moving 0-2pi
    # theta : 0 to pi
    # phi : à to 2*pi


def mie(nb_angles, mu, mu0, indx, r, lambda0):
    """
    Computation of MIE phase function for given arguments.
    
    Args:
        nb_angles: nb of positives angles (from 0 to 1) = len(mu)/2
        mu: list of cosines of computaion angles
        mu0: cosine of incidence angle
        indx: refractive index of the sphere (complex)
        r: sphere radius (in m)
        lambda0: wavelength (in m)
        
    Returns:
        matrix of phase functions P0=P(µ0, µ) and P=(µ',µ)

    Estimated time of computing : 11 minutes
    """    
    
    nb_phi = 25
    phi = np.linspace(0, np.pi, nb_phi)
    phi0 = 0

    # Computing size parameter
    x = 2*np.pi*r/lambda0

    # Phase function for 1st order of scattering : P(µ, µ0)
    P0 = np.zeros(2*nb_angles)

    # Vectorized computation for P(µ, µ0)
    for m in tqdm(range(2*nb_angles), desc="P(µ, µ0)", leave=True):
        # Compute all mu_diff for this m at once
        mu_diff_pos = - (mu[m]*mu0 + np.sqrt(1-mu0*mu0)*np.sqrt(1-mu[m]*mu[m])*np.cos(phi0-phi))
        mu_diff_neg = - (mu[m]*mu0 - np.sqrt(1-mu0*mu0)*np.sqrt(1-mu[m]*mu[m])*np.cos(phi0-phi))
        
        # Vectorized phase function computation
        P_pos = miepython.i_unpolarized(indx, x, mu_diff_pos)
        P_neg = miepython.i_unpolarized(indx, x, mu_diff_neg)
        
        P0[m] = np.trapz(P_pos + P_neg, phi)/(4*np.pi)

    P0 = P0/np.trapz(P0, mu) * 2  # normalization

    # Phase function for n-th order of scattering : P(µ, µ')
    P = np.zeros((2*nb_angles, 2*nb_angles))

    # Pre-compute all unique mu_diff values to avoid redundant calculations
    unique_mu_diffs = set()
    mu_diff_map = {}  # Cache for computed phase function values
    
    for n in tqdm(range(2*nb_angles), desc="Diffusion angles", leave=True):
        for m in range(n, 2*nb_angles):
            cosm_cosn = mu[m]*mu[n]
            sinm_sinn = np.sqrt(1-mu[n]*mu[n])*np.sqrt(1-mu[m]*mu[m])
            
            for p in range(nb_phi):
                mu_diff_pos = - (cosm_cosn + sinm_sinn*np.cos(phi0-phi[p]))
                mu_diff_neg = - (cosm_cosn - sinm_sinn*np.cos(phi0-phi[p]))
                unique_mu_diffs.add(round(mu_diff_pos, 6))
                unique_mu_diffs.add(round(mu_diff_neg, 6))
    
    # Pre-compute phase function for all unique mu_diff values
    unique_mu_diffs = np.array(list(unique_mu_diffs))
    print(f"Pre-computing phase function for {len(unique_mu_diffs)} unique angles...")
    phase_values = miepython.i_unpolarized(indx, x, unique_mu_diffs)
    
    # Create lookup dictionary
    for i, mu_diff in enumerate(unique_mu_diffs):
        mu_diff_map[round(mu_diff, 6)] = phase_values[i]

    # Now compute P matrix using cached values
    for n in tqdm(range(2*nb_angles), desc="P(µ, µ')", leave=True):
        for m in range(n, 2*nb_angles):
            cosm_cosn = mu[m]*mu[n]
            sinm_sinn = np.sqrt(1-mu[n]*mu[n])*np.sqrt(1-mu[m]*mu[m])
            
            P_pos = np.zeros(nb_phi)
            P_neg = np.zeros(nb_phi)
            
            for p in range(nb_phi):
                mu_diff_pos = - (cosm_cosn + sinm_sinn*np.cos(phi0-phi[p]))
                mu_diff_neg = - (cosm_cosn - sinm_sinn*np.cos(phi0-phi[p]))
                
                # Use cached values
                P_pos[p] = mu_diff_map[round(mu_diff_pos, 6)]
                P_neg[p] = mu_diff_map[round(mu_diff_neg, 6)]

            P[m,n] = np.trapz(P_pos + P_neg, phi)/(2*np.pi)
            P[n,m] = P[m,n]  # symmetry

        P[:,n] = 4 * P[:,n]/(np.trapz(P[:,n], mu[:]))  # normalization
    
    return P0, P

    # incidence angle on a circle : location of the sun is graphed by phi moving 0-2pi
    # theta : 0 to pi
    # phi : à to 2*pi



def log_normal_mie(nb_angles, mu, mu0, wl, idx, N0, r_m, sig, phase_fun):

    mu_neg = np.linspace(-1,0, nb_angles)
    mu_pos = np.linspace(0,1, nb_angles)
    mu = np.concatenate((mu_neg, mu_pos))

    nb_radius = 100
    list_radius = np.linspace(0.01, 10, nb_radius)

    # Size distribution
    print('Compute size distribution n(r)')
    coeff_norm = N0 / (np.sqrt(2*np.pi) * np.log(sig))
    coeff_exp = 2*(np.log(sig)**2)
    n_r = (1 / list_radius) * np.exp( - ((np.log(list_radius)-np.log(r_m))**2) / coeff_exp) # coeff_norm omitted as phase function is normalized


    print('======================================================')
    print("         Pre-computing Qsca from Mie theory")
    print('======================================================')

    x_list = 2 * np.pi * list_radius / wl
    _, Qsca, _, _ = miepython.efficiencies(idx, x_list, wl) 
    coef_int = n_r * Qsca
    denom = np.trapz(coef_int, list_radius)

    nb_phi = 25
    phi = np.linspace(0, np.pi, nb_phi)
    phi0=0
    
    mu_diff_list, P_list = None, None


    print('======================================================')
    print("         Computing phase function P(µ, µ0)")
    print('======================================================')


    folder = fr'D:\Polytechnique\4_3A\STAGE_3A\Harvard\SOS DYKEMA\Code perso\SOS_AER'
    os.makedirs(folder, exist_ok=True)
    if phase_fun == 'eva':
        filename = f'EVA_P0={mu0}.txt'
    else:
        filename = f'WF_P0={mu0}.txt'
    full_path = os.path.join(folder, filename)

    skip_P0 = False
    if os.path.exists(full_path):
        with open(full_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            mu0_file = None
            nb_angles_file = None
            nb_phi_file = None
            for line in lines:
                if line.startswith('mu0'):
                    mu0_file = float(line.split('=')[1].strip())
                if line.startswith('nb_angles'):
                    nb_angles_file = int(line.split('=')[1].strip())
                if line.startswith('nb_phi'):
                    nb_phi_file = int(line.split('=')[1].strip())
            if nb_angles_file == nb_angles and nb_phi_file == nb_phi and mu0_file == mu0:
                print(f"P(µ, µ0) already exists with matching parameters, skipping computation.")
                skip_P0 = True
    if not skip_P0:

        print("File not found, computing P(µ, µ0)")

        if mu_diff_list == None or P_list == None:
            mu_diff_list, P_list = compute_P(x_list, idx)

        sqrt1mu02 = np.sqrt(1 - mu0**2)
        I_vals_pos = np.zeros_like(x_list)
        I_vals_neg = np.zeros_like(x_list)
        sqrt1mu2 = np.sqrt(1 - mu**2)

        P_pos = np.zeros(nb_phi)
        P_neg = np.zeros(nb_phi)
        P0 = np.zeros(2*nb_angles)

        for m in tqdm(range(2*nb_angles), desc='Computing P0 for mu0=' + str(mu0)):

            mu_diff_pos = - (mu[m]*mu0 + sqrt1mu02*sqrt1mu2[m]*np.cos(phi0-phi))
            mu_diff_neg = - (mu[m]*mu0 - sqrt1mu02*sqrt1mu2[m]*np.cos(phi0-phi))

            for p in range(nb_phi):
                    
                for i, x in enumerate(x_list):
                    I_vals_pos[i] = interpolate_phase(mu_diff_list, P_list[i,:], mu_diff_pos[p])
                    I_vals_neg[i] = interpolate_phase(mu_diff_list, P_list[i,:], mu_diff_neg[p])
                P_pos[p] = np.trapz(coef_int[:] * I_vals_pos, list_radius)
                P_neg[p] = np.trapz(coef_int[:] * I_vals_neg, list_radius)

            P0[m] = np.trapz(P_pos + P_neg, phi)/(4*np.pi)

        P0 = P0/np.trapz(P0, mu) * 2  # normalization

        
        with open(full_path, 'w', encoding='utf-8') as f:
            f.write(f'mu0 = {mu0}\n')
            f.write(f'nb_angles = {nb_angles}\n')
            f.write(f'nb_phi = {nb_phi}\n')
            f.write('mu' + "'" + '\tP(µ, µ0)\n')
            for m in range(2*nb_angles):
                f.write(f'{mu[m]}\t{P0[m]}\n')
        print(f"P(µ, µ0) saved in {full_path}")

    # Read the P(µ, µ0) file
    mu_file = []
    with open(full_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    for i, line in enumerate(lines):
        if line.startswith('mu0'):
            mu0_file = float(line.split('=')[1].strip())
        if line.startswith('nb_angles'):
            nb_angles_file = int(line.split('=')[1].strip())
        if line.startswith('nb_phi'):
            nb_phi_file = int(line.split('=')[1].strip())
        if line.startswith('mu') and "P(µ, µ0)" in line:
            data_start = i + 1
            break
    P0 = np.zeros(2*nb_angles)
    for line in lines[data_start:]:
        if line.strip() == "" or line.startswith('mu'):
            continue
        parts = line.strip().split('\t')
        if len(parts) < 2:
            continue
        mu1 = float(parts[0])
        P0val = float(parts[1])
        # We assume mu is symmetric and ordered
        if len(mu_file) < 2*nb_angles:
            mu_file.append(mu1)
        m = mu_file.index(mu1) if mu1 in mu_file else len(mu_file)-1
        P0[m] = P0val


    print('======================================================')
    print("         Computing phase function P(µ, µ')")
    print('======================================================')

    folder = fr'D:\Polytechnique\4_3A\STAGE_3A\Harvard\SOS DYKEMA\Code perso\SOS_AER'
    os.makedirs(folder, exist_ok=True)
    if phase_fun == 'eva':
        filename = f'EVA_P.txt'
    else:
        filename = f'WF_P.txt'
    full_path = os.path.join(folder, filename)

    skip_P = False
    if os.path.exists(full_path):
        with open(full_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            nb_angles_file = None
            nb_phi_file = None
            for line in lines:
                if line.startswith('nb_angles'):
                    nb_angles_file = int(line.split('=')[1].strip())
                if line.startswith('nb_phi'):
                    nb_phi_file = int(line.split('=')[1].strip())
            if nb_angles_file == nb_angles and nb_phi_file == nb_phi:
                print(f"P(µ, µ') already exists with matching parameters, skipping computation.")
                skip_P = True
    if not skip_P:

        print("File not found, computing P(µ, µ')")

        if mu_diff_list == None or P_list == None:
            mu_diff_list, P_list = compute_P(x_list, idx)

        P = np.zeros((2*nb_angles, 2*nb_angles))

        # Prepare arguments for multiprocessing
        args_list = [(n, mu, mu_diff_list, P_list, nb_angles, phi, phi0, nb_phi, x_list, coef_int, list_radius) 
                    for n in range(2*nb_angles)]

        # Use multiprocessing to compute columns in parallel
        with Pool(processes=min(cpu_count(), 8)) as pool:
            results = list(tqdm(pool.imap(compute_P_column, args_list), 
                              total=2*nb_angles, desc='Computing P columns'))

        # Collect results and assemble the matrix
        for n, P_col in results:
            P[:, n] = P_col

        # Apply symmetry: P[m,n] = P[n,m] for all m, n
        print("Applying symmetry...")
        for m in range(2*nb_angles):
            for n in range(m, 2*nb_angles):  # Only for n < m to avoid double copying
                P[m, n] = P[n, m]

        # Normalization
        for n in range(2*nb_angles):
            norm_factor = np.trapz(P[:,n], mu[:])
            if norm_factor > 1e-10:  # Avoid division by zero
                P[:,n] = 4 * P[:,n] / norm_factor  # normalization
            else:
                print(f"Warning: Very small normalization factor for column {n}: {norm_factor}")

        # Debug: check that we have non-zero values everywhere
        non_zero_count = np.count_nonzero(P)
        total_elements = P.size
        zero_count = total_elements - non_zero_count
        print(f"Matrix P: {non_zero_count}/{total_elements} non-zero elements")
        print(f"P diagonal values (m=n): {np.diag(P)[:5]}")  # First 5 diagonal values
        print(f"P off-diagonal values (m≠n): {P[0,1], P[1,2], P[2,3]}")  # Some off-diagonal values
        
        # Check for zero values and show warning
        if zero_count > 0:
            print(f"    WARNING: {zero_count} zero values found in matrix P!")
            print(f"   Zero percentage: {zero_count/total_elements*100:.2f}%")
            
            # Find positions of some zero values
            zero_positions = np.where(P == 0.0)
            if len(zero_positions[0]) > 0:
                print(f"   First few zero positions: (m,n) = {list(zip(zero_positions[0][:5], zero_positions[1][:5]))}")
        else:
            print("  SUCCESS: No zero values found in matrix P!")
        
        # Additional analysis: check if zeros are in specific patterns
        if zero_count > 0:
            # Check if zeros are only in the lower triangle (before symmetry application)
            lower_triangle_zeros = 0
            for m in range(2*nb_angles):
                for n in range(m):
                    if P[m, n] == 0.0:
                        lower_triangle_zeros += 1
            
            if lower_triangle_zeros == zero_count:
                print("   Note: All zeros are in the lower triangle - this is expected before symmetry application")
            else:
                print("     WARNING: Zeros found outside the lower triangle - this indicates a problem!")

        
        with open(full_path, 'w', encoding='utf-8') as f:
            f.write(f'nb_angles = {nb_angles}\n')
            f.write(f'nb_phi = {nb_phi}\n')
            f.write('mu\tmu' + "'" + '\tP(µ1, µ2)\n')
            for m in range(2*nb_angles):
                for n in range(2*nb_angles):  # Write all values (complete matrix)
                    f.write(f'{mu[m]}\t{mu[n]}\t{P[m,n]}\n')
        print(f"P(µ, µ') saved in {full_path}")

    
    

    # Read the P(µ, µ') file
    mu_file = []
    with open(full_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    for i, line in enumerate(lines):
        if line.startswith('nb_angles'):
            nb_angles_file = int(line.split('=')[1].strip())
        if line.startswith('nb_phi'):
            nb_phi_file = int(line.split('=')[1].strip())
        if line.startswith('mu') and "P(µ1, µ2)" in line:
            data_start = i + 1
            break
    P = np.zeros((2*nb_angles, 2*nb_angles))
    for line in lines[data_start:]:
        if line.strip() == "" or line.startswith('mu'):
            continue
        parts = line.strip().split('\t')
        if len(parts) < 3:
            continue
        mu1 = float(parts[0])
        mu2 = float(parts[1])
        Pval = float(parts[2])
        # We assume mu is symmetric and ordered
        if len(mu_file) < 2*nb_angles:
            mu_file.append(mu1)
        m = mu_file.index(mu1) if mu1 in mu_file else len(mu_file)-1
        n = mu_file.index(mu2) if mu2 in mu_file else len(mu_file)-1
        P[m, n] = Pval
        P[n, m] = Pval  # symmetry
    
    if P.shape != (2*nb_angles, 2*nb_angles): 
        print(f'Error P dimension: got {P.shape}, expected ({2*nb_angles}, {2*nb_angles})')
    if P0.shape != (2*nb_angles,): 
        print(f'Error P0 dimension: got {P0.shape}, expected ({2*nb_angles},)')
    
    P0 = P0/np.trapz(P0, mu) *2
    for n in range(2*nb_angles):
        P[:,n] = 4 * P[:,n]/(np.trapz(P[:,n], mu[:]))
    
    return P0, P


def compute_P(x_list, idx):
    nb_angles_diff = 6001
    mu_diff_list = np.linspace(-1, 1, nb_angles_diff)
    nb_radius = len(x_list)
    if 0 not in mu_diff_list:
        print('Error: 0 is not in mu_diff_list')
    P_list = np.zeros((len(x_list), nb_angles_diff))
    for i, x in enumerate(x_list):
        for m in tqdm(range(nb_angles_diff), desc=f'Index n={i+1}/{nb_radius}'):
            P_list[i, m] = miepython.i_unpolarized(idx, x, mu_diff_list[m])
    return mu_diff_list, P_list

def interpolate_phase(mu_diff_list, P_list, mu_diff):
    if mu_diff<-1 or mu_diff>1:
        print(f'Error: diffusion angle is out of range (mu_diff={mu_diff}), using np.clip')
    mu_diff = np.clip(mu_diff, -1, 1)
    idx = np.searchsorted(mu_diff_list, mu_diff)
    if idx == 0:
        return P_list[0]
    elif idx >= len(mu_diff_list):
        return P_list[-1]
    mu_low = mu_diff_list[idx-1]
    mu_high = mu_diff_list[idx]
    P_low = P_list[idx-1]
    P_high = P_list[idx]
    weight = (mu_diff - mu_low) / (mu_high - mu_low)
    P_interpolated = P_low + weight * (P_high - P_low)
    return P_interpolated

def compute_P_column(args):
    """
    Compute one column of the P matrix for multiprocessing compatibility.
    Returns (n, P_col) where P_col is the computed column.
    """
    n, mu, mu_diff_list, P_list, nb_angles, phi, phi0, nb_phi, x_list, coef_int, list_radius = args
    
    sin_n = np.sqrt(1-mu[n]*mu[n])
    P_col = np.zeros(2*nb_angles)
    
    # Calculate only for m >= n (upper triangle) - this is the optimization!
    for m in range(n, 2*nb_angles):
        cosm_cosn = mu[n]*mu[m]
        sinm_sinn = sin_n*np.sqrt(1-mu[m]*mu[m])

        mu_diff_pos = - (cosm_cosn + sinm_sinn*np.cos(phi0-phi))
        mu_diff_neg = - (cosm_cosn - sinm_sinn*np.cos(phi0-phi))
        
        # Ensure mu_diff values are within [-1, 1]
        mu_diff_pos = np.clip(mu_diff_pos, -1, 1)
        mu_diff_neg = np.clip(mu_diff_neg, -1, 1)
        
        P_pos = np.zeros(nb_phi)
        P_neg = np.zeros(nb_phi)

        I_vals_pos = np.zeros_like(x_list)
        I_vals_neg = np.zeros_like(x_list)
        
        for p in range(nb_phi):
            for i, x in enumerate(x_list):
                I_vals_pos[i] = interpolate_phase(mu_diff_list, P_list[i,:], mu_diff_pos[p])
                I_vals_neg[i] = interpolate_phase(mu_diff_list, P_list[i,:], mu_diff_neg[p])
            P_pos[p] = np.trapz(coef_int[:] * I_vals_pos, list_radius)
            P_neg[p] = np.trapz(coef_int[:] * I_vals_neg, list_radius)
        
        P_col[m] = np.trapz(P_pos+P_neg, phi)/(2*np.pi)
    
    # For m < n, we'll copy from the symmetric position later
    # This will be handled in the main loop after all columns are computed
    
    return n, P_col

