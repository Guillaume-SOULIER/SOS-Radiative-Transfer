import numpy as np
import matplotlib.pyplot as plt
import copy
import os
from tqdm import tqdm
import warnings
from I1_In import I1_NumInt, Jn_NumInt, In_NumInt, mu_approx_In
from SOS_Aer_global_va import MU_THRESHOLD, MU_EXTREME_THRESHOLD, MU_VERY_SMALL_THRESHOLD
from SOS_Aer_phase_func import phase_func
from SOS_Aer_tau_profile import tau_profile
from SOS_Aer_In_limit import asymptotic_downward_radiance, improved_asymptotic_downward_radiance, improved_limit_mu_down, doubling_method_downward_radiance, mu_approx_In, limit_mu_down

# Suppress deprecation warnings for np.trapz
warnings.simplefilter('ignore', DeprecationWarning)


################################################ 
#           Compute radiative forcing
################################################ 
def SOS_Aer_radiative_forcing(tauStar_aer, dtau_aer, tauStar_atm, dtau_atm, P_aer, P0_aer, alb_aer, P_atm, P0_atm, alb_atm, grd_alb, F0, mu, mu0, nb_angles, tau, nb_layers, idx_up, idx_down):

    # 1st order of scattering
    I1 = np.zeros((nb_layers, 2*nb_angles))
    print('Computing radiance fields for n=1: In/I = -')

    # Compute downward field for upper atmospheric layer
    m_arr = np.arange(nb_angles-1) # [0, ..., nb_angles-2]
    mask_mu0 = np.abs(mu[m_arr] + mu0) < 0.0001 # angles |µ+µ0|<0.0001
    
    for t in range(idx_up):        
        ## general case: µ != 0 and µ0
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            # Already scattered
            scatt_before = np.zeros(nb_angles-1)
            # Scattered before reaching the surface
            scatt_direct = (mu0/(mu0+mu[m_arr])) * (alb_atm/(4*np.pi)) * P0_atm[m_arr] * F0 * (np.exp(-tau[t] / mu0) - np.exp(+tau[t]/mu[m_arr]))
            # Scattered after reflecting on the surface
            scatt_surface = (mu0/(mu0-mu[m_arr])) * (alb_atm/(4*np.pi)) * P0_atm[2*nb_angles-1-m_arr] * (F0*grd_alb*np.exp(-tauStar_tot/mu0)) * (np.exp(-(tauStar_tot-tau[t])/mu0)-np.exp(-(tauStar_tot)/mu0)*np.exp(tau[t]/mu[m_arr]))
            I1[t, m_arr] =  scatt_before + scatt_direct + scatt_surface
        ## case µ = 0
        # Alreday scattered
        scatt_before = 0
        # Scattered before reaching the surface
        scatt_direct = (alb_atm/(4*np.pi)) * (mu0/(mu0+mu[nb_angles-1])) * P0_atm[nb_angles-1] * F0 * np.exp(-tau[t] / mu0)
        # Scattered after reflecting on the surface
        scatt_surface = (mu0/(mu0-mu[nb_angles-1])) * (alb_atm/(4*np.pi)) * P0_atm[2*nb_angles-1-(nb_angles-1)] * (F0*grd_alb*np.exp(-tauStar_tot/mu0)) * np.exp(-(tauStar_tot-tau[t])/mu0)
        I1[t, nb_angles-1] = scatt_before + scatt_direct + scatt_surface
        # Case |µ|=µ0        
        if np.any(mask_mu0):
            # Alreday scattered
            scatt_before = np.zeros(len(m_arr[mask_mu0]))
            # Scattered before reaching the surface
            scatt_direct = (alb_atm/(4*np.pi)) * P0_atm[m_arr[mask_mu0]] * F0 * np.exp(-tau[t] / mu0) * tau[t] / mu0
            # Scattered after reflecting on the surface
            scatt_surface = (mu0/(mu0-mu[m_arr[mask_mu0]])) * (alb_atm/(4*np.pi)) * P0_atm[2*nb_angles-1-m_arr[mask_mu0]] * (F0*grd_alb*np.exp(-tauStar_tot/mu0)) * (np.exp(-(tauStar_tot-tau[t])/mu0)-np.exp(-(tauStar_tot)/mu0)*np.exp(tau[t]/mu[m_arr[mask_mu0]]))
            I1[t, m_arr[mask_mu0]] = scatt_before + scatt_direct + scatt_surface

    for t in range(idx_up, idx_down+1):
        # general case: µ != 0 and µ0
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            # Already scattered
            scatt_before = I1[idx_up-1, m_arr] * np.exp((tau[t]-tau[idx_up-1])/mu[m_arr])
            # Scattered before reaching the surface
            scatt_direct = (mu0/(mu0+mu[m_arr])) * (alb_atm * P0_atm[m_arr] * (dtau_atm/(dtau_atm+dtau_aer)) + alb_aer * P0_aer[m_arr] * (dtau_aer/(dtau_atm+dtau_aer))) * (F0/(4*np.pi)) * (np.exp(-tau[t] / mu0) - np.exp(-tau[idx_up-1] / mu0) * np.exp(+(tau[t]-tau[idx_up-1])/mu[m_arr]))
            # Scattered after reflecting on the surface
            scatt_surface = (mu0/(mu0-mu[m_arr])) * (alb_atm * P0_atm[2*nb_angles-1-m_arr] * (dtau_atm/(dtau_atm+dtau_aer)) + alb_aer * P0_aer[2*nb_angles-1-m_arr] * (dtau_aer/(dtau_atm+dtau_aer))) * (F0*grd_alb*np.exp(-tauStar_tot/mu0)/(4*np.pi)) * (np.exp(-(tauStar_tot-tau[t])/mu0)-np.exp(-(tauStar_tot-tau[idx_up])/mu0)*np.exp(+(tau[t]-tau[idx_up])/mu[m_arr]))
            I1[t, m_arr] =  scatt_before + scatt_direct + scatt_surface
        ## case µ = 0
        # Alreday scattered
        scatt_before = 0
        # Scattered before reaching the surface
        scatt_direct = (mu0/(mu0+mu[nb_angles-1])) * (alb_atm * P0_atm[nb_angles-1] * (dtau_atm/(dtau_atm+dtau_aer)) + alb_aer * P0_aer[nb_angles-1] * (dtau_aer/(dtau_atm+dtau_aer))) * (F0/(4*np.pi)) * np.exp(-tau[t] / mu0)
        # Scattered after reaching the surface
        scatt_surface = (mu0/(mu0-mu[nb_angles-1])) * (alb_atm * P0_atm[2*nb_angles-1-(nb_angles-1)] * (dtau_atm/(dtau_atm+dtau_aer)) + alb_aer * P0_aer[2*nb_angles-1-(nb_angles-1)] * (dtau_aer/(dtau_atm+dtau_aer))) * (F0*grd_alb*np.exp(-tauStar_tot/mu0)/(4*np.pi)) * np.exp(-(tauStar_tot-tau[t])/mu0)
        I1[t, nb_angles-1] = scatt_before + scatt_direct + scatt_surface
        # Case |µ|=µ0
        if np.any(mask_mu0):
            # Alreday scattered
            scatt_before = I1[idx_up-1, m_arr[mask_mu0]] * np.exp((tau[t]-tau[idx_up-1])/mu[m_arr[mask_mu0]])
            # Scattered before reaching the surface
            scatt_direct = (alb_atm * P0_atm[m_arr[mask_mu0]] * (dtau_atm/(dtau_atm+dtau_aer)) + alb_aer * P0_aer[m_arr[mask_mu0]] * (dtau_aer/(dtau_atm+dtau_aer))) * (F0/(4*np.pi)) * np.exp(-tau[t] / mu0) * (tau[t]-tau[idx_up-1]) / mu0
            # Scatterd after reflecting on the surface
            scatt_surface = (mu0/(mu0-mu[m_arr[mask_mu0]])) * (alb_atm * P0_atm[2*nb_angles-1-m_arr[mask_mu0]] * (dtau_atm/(dtau_atm+dtau_aer)) + alb_aer * P0_aer[2*nb_angles-1-m_arr[mask_mu0]] * (dtau_aer/(dtau_atm+dtau_aer))) * (F0*grd_alb*np.exp(-tauStar_tot/mu0)/(4*np.pi)) * (np.exp(-(tauStar_tot-tau[t])/mu0)-np.exp(-(tauStar_tot-tau[idx_up])/mu0)*np.exp(+(tau[t]-tau[idx_up])/mu[m_arr[mask_mu0]]))
            I1[t, m_arr[mask_mu0]] = scatt_before + scatt_direct + scatt_surface

    for t in range(idx_down+1, nb_layers):
        # general case: µ != 0 and µ0
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            # Already scattered
            scatt_before = I1[idx_down, m_arr] * np.exp((tau[t]-tau[idx_down])/mu[m_arr])
            # Scattered before reaching the surface
            scatt_direct = (mu0/(mu0+mu[m_arr])) * (alb_atm/(4*np.pi)) * P0_atm[m_arr] * F0 * (np.exp(-tau[t] / mu0) - np.exp(-tau[idx_down] / mu0) * np.exp(+(tau[t]-tau[idx_down])/mu[m_arr]))
            # Scattered after reflecting on the surface
            scatt_surface = (mu0/(mu0-mu[m_arr])) * (alb_atm/(4*np.pi)) * P0_atm[2*nb_angles-1-m_arr] * (F0*grd_alb*np.exp(-tauStar_tot/mu0)) * (np.exp(-(tauStar_tot-tau[t])/mu0)-np.exp(-(tauStar_tot-tau[idx_down+1])/mu0)*np.exp(+(tau[t]-tau[idx_down+1])/mu[m_arr]))
            I1[t, m_arr] = scatt_before + scatt_direct + scatt_surface
        ## case µ = 0
        # Alreday scattered
        scatt_before = 0
        # Scattered before reaching the surface
        scatt_direct = (alb_atm/(4*np.pi)) * (mu0/(mu0+mu[nb_angles-1])) * P0_atm[nb_angles-1] * F0 * np.exp(-tau[t] / mu0)
        # Scattered after reflecting on the surface
        scatt_surface = (mu0/(mu0-mu[nb_angles-1])) * (alb_atm/(4*np.pi)) * P0_atm[2*nb_angles-1-(nb_angles-1)] * (F0*grd_alb*np.exp(-tauStar_tot/mu0)) * np.exp(-(tauStar_tot-tau[t])/mu0)
        I1[t, nb_angles-1] = scatt_before + scatt_direct + scatt_surface
        # Case |µ|=µ0
        if np.any(mask_mu0):
            # Alreday scattered
            scatt_before = I1[idx_down, m_arr[mask_mu0]] * np.exp((tau[t]-tau[idx_down])/mu[m_arr[mask_mu0]])
            # Scattered before reaching the surface
            scatt_direct = (alb_atm/(4*np.pi)) * P0_atm[m_arr[mask_mu0]] * F0 * np.exp(-tau[t] / mu0) * (tau[t]-tau[idx_down]) / mu0
            # Scatterd after reflecting on the surface
            scatt_surface = (mu0/(mu0-mu[m_arr[mask_mu0]])) * (alb_atm/(4*np.pi)) * P0_atm[2*nb_angles-1-m_arr[mask_mu0]] * (F0*grd_alb*np.exp(-tauStar_tot/mu0)) * (np.exp(-(tauStar_tot-tau[t])/mu0)-np.exp(-(tauStar_tot-tau[idx_down+1])/mu0)*np.exp(+(tau[t]-tau[idx_down+1])/mu[m_arr[mask_mu0]]))
            I1[t, m_arr[mask_mu0]] = scatt_before + scatt_direct + scatt_surface


    
    # Compute upward field for upper atmospheric layer
    m_arr = np.arange(nb_angles+1, 2*nb_angles)
    mask_mu0 = np.abs(mu[m_arr] - mu0) < 0.0001 # angles |µ-µ0|<0.0001

    for t in range(idx_down+1, nb_layers):
        ## general case: µ != 0 and µ0
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            # Already scattered
            scatt_before = grd_alb * I1[nb_layers-1, nb_angles-1-(m_arr-nb_angles)] * np.exp(-(tau[nb_layers-1]-tau[t])/mu[m_arr])
            # Scattered before reaching the surface
            scatt_direct = (mu0/(mu0+mu[m_arr])) * (alb_atm/(4*np.pi)) * P0_atm[m_arr] * F0 * (np.exp(-tau[t] / mu0) - np.exp(-tau[nb_layers-1] / mu0) * np.exp(-(tau[nb_layers-1]-tau[t])/mu[m_arr])) 
            # Scattered after reflecting on the ground
            scatt_surface = (mu0/(mu0-mu[m_arr])) * (alb_atm/(4*np.pi)) * P0_atm[2*nb_angles-1-m_arr] * (F0*grd_alb*np.exp(-tauStar_tot/mu0)) * (np.exp(-(tauStar_tot-tau[t])/mu0) - np.exp(-(tauStar_tot-tau[t])/mu[m_arr]))
            I1[t, m_arr] = scatt_before + scatt_direct + scatt_surface
        ## case µ = 0
        # Alreday scattered
        scatt_before = 0
        # Scattered before reaching the surface
        scatt_direct = (alb_atm/(4*np.pi)) * (mu0/(mu0+mu[nb_angles])) * P0_atm[nb_angles] * F0 * np.exp(-tau[t] / mu0)
        # Scattered after reflecting on the surface
        scatt_surface = (mu0/(mu0-mu[nb_angles])) * (alb_atm/(4*np.pi)) * P0_atm[2*nb_angles-1-nb_angles] * (F0*grd_alb*np.exp(-tauStar_tot/mu0)) * np.exp(-(tauStar_tot-tau[t])/mu0) 
        I1[t, nb_angles] = scatt_before + scatt_direct + scatt_surface
        ## Case µ=µ0
        if np.any(mask_mu0):
            # Alreday scattered
            scatt_before = grd_alb * I1[nb_layers-1, nb_angles-1-(m_arr[mask_mu0]-nb_angles)] * np.exp(-(tau[nb_layers-1]-tau[t])/mu[m_arr[mask_mu0]])
            # Scattered before reaching the surface
            scatt_direct = (mu0/(mu0+mu[m_arr[mask_mu0]])) * (alb_atm/(4*np.pi)) * P0_atm[m_arr[mask_mu0]] * F0 * (np.exp(-tau[t] / mu0) - np.exp(-tau[nb_layers-1] / mu0) * np.exp(-(tau[nb_layers-1]-tau[t])/mu[m_arr[mask_mu0]])) 
            # Scatterd after reflecting on the surface
            scatt_surface = (alb_atm/(4*np.pi)) * P0_atm[2*nb_angles-1-m_arr[mask_mu0]] * (F0*grd_alb*np.exp(-tauStar_tot/mu0)) * np.exp(-(tauStar_tot-tau[t])/mu0) * (tauStar_tot-tau[t]) / mu0
            I1[t, m_arr[mask_mu0]] = scatt_before + scatt_direct + scatt_surface
        
    for t in range(idx_up, idx_down+1):
        # general case: µ != 0 and µ0
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            # Already scattared
            scatt_before = I1[idx_down+1, m_arr] * np.exp(-(tau[idx_down+1]-tau[t])/mu[m_arr])
            # Scattered before reaching the surface
            scatt_direct = (mu0/(mu0+mu[m_arr])) * (alb_atm * P0_atm[m_arr] * (dtau_atm/(dtau_atm+dtau_aer)) + alb_aer * P0_aer[m_arr] * (dtau_aer/(dtau_atm+dtau_aer))) * (F0/(4*np.pi)) * (np.exp(-tau[t] / mu0) - np.exp(-tau[idx_down+1] / mu0) * np.exp(-(tau[idx_down+1]-tau[t])/mu[m_arr]))
            # Scattered after reflecting on the ground
            scatt_surface = (mu0/(mu0-mu[m_arr])) * (alb_atm * P0_atm[2*nb_angles-1-m_arr] * (dtau_atm/(dtau_atm+dtau_aer)) + alb_aer * P0_aer[2*nb_angles-1-m_arr] * (dtau_aer/(dtau_atm+dtau_aer))) * (F0*grd_alb*np.exp(-tauStar_tot/mu0)/(4*np.pi)) * (np.exp(-(tauStar_tot-tau[t])/mu0) - np.exp(-(tauStar_tot-tau[idx_down])/mu0)*np.exp(-(tau[idx_down]-tau[t])/mu[m_arr]))
            I1[t, m_arr] = scatt_before + scatt_direct + scatt_surface
        ## case µ = 0
        # Alreday scattered
        scatt_before = 0
        # Scattered before reaching the surface
        scatt_direct = (mu0/(mu0+mu[nb_angles])) * (alb_atm * P0_atm[nb_angles] * (dtau_atm/(dtau_atm+dtau_aer)) + alb_aer * P0_aer[nb_angles] * (dtau_aer/(dtau_atm+dtau_aer))) * (F0/(4*np.pi)) * np.exp(-tau[t] / mu0)
        # Scattered after reflecting on the surface
        scatt_surface = (mu0/(mu0-mu[nb_angles])) * (alb_atm * P0_atm[2*nb_angles-1-nb_angles] * (dtau_atm/(dtau_atm+dtau_aer)) + alb_aer * P0_aer[2*nb_angles-1-nb_angles] * (dtau_aer/(dtau_atm+dtau_aer))) * (F0*grd_alb*np.exp(-tauStar_tot/mu0)/(4*np.pi)) * np.exp(-(tauStar_tot-tau[t])/mu0)
        I1[t, nb_angles] = scatt_before + scatt_direct + scatt_surface
        ## Case µ=µ0
        if np.any(mask_mu0):
            # Alreday scattered
            scatt_before = I1[idx_down+1, m_arr[mask_mu0]] * np.exp(-(tau[idx_down+1]-tau[t])/mu[m_arr[mask_mu0]])
            # Scattered before reaching the surface
            scatt_direct = (mu0/(mu0+mu[m_arr[mask_mu0]])) * (alb_atm * P0_atm[m_arr[mask_mu0]] * (dtau_atm/(dtau_atm+dtau_aer)) + alb_aer * P0_aer[m_arr[mask_mu0]] * (dtau_aer/(dtau_atm+dtau_aer))) * (F0/(4*np.pi)) * (np.exp(-tau[t] / mu0) - np.exp(-tau[idx_down+1] / mu0) * np.exp(-(tau[idx_down+1]-tau[t])/mu[m_arr[mask_mu0]]))
            # Scattered after reflecting on the ground
            scatt_surface = (alb_atm * P0_atm[2*nb_angles-1-m_arr[mask_mu0]] * (dtau_atm/(dtau_atm+dtau_aer)) + alb_aer * P0_aer[2*nb_angles-1-m_arr[mask_mu0]] * (dtau_aer/(dtau_atm+dtau_aer))) * (F0*grd_alb*np.exp(-tauStar_tot/mu0)/(4*np.pi)) * np.exp(-(tauStar_tot-tau[t])/mu0) * (tau[idx_down]-tau[t]) / mu0
            I1[t, m_arr[mask_mu0]] = scatt_before + scatt_direct + scatt_surface
        
    
    for t in range(idx_up):
        ## general case: µ != 0 and µ0
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            # Already scattered
            scatt_before = I1[idx_up, m_arr] * np.exp(-(tau[idx_up]-tau[t])/mu[m_arr])
            # Scattered before reaching the surface
            scatt_direct = (mu0/(mu0+mu[m_arr])) * (alb_atm/(4*np.pi)) * P0_atm[m_arr] * F0 * (np.exp(-tau[t] / mu0) - np.exp(-tau[idx_up] / mu0) * np.exp(-(tau[idx_up]-tau[t])/mu[m_arr]))
            # Scattered after reflecting on the surface
            scatt_surface = (mu0/(mu0-mu[m_arr])) * (alb_atm/(4*np.pi)) * P0_atm[2*nb_angles-1-m_arr] * (F0*grd_alb*np.exp(-tauStar_tot/mu0)) * (np.exp(-(tauStar_tot-tau[t])/mu0) - np.exp(-(tauStar_tot-tau[idx_up-1])/mu0)*np.exp(-(tau[idx_up-1]-tau[t])/mu[m_arr]))
            I1[t, m_arr] = scatt_before + scatt_direct + scatt_surface
        ## case µ = 0
        # Alreday scattered
        scatt_before = 0
        # Scattered before reaching the surface
        scatt_direct = (alb_atm/(4*np.pi)) * (mu0/(mu0+mu[nb_angles])) * P0_atm[nb_angles] * F0 * np.exp(-tau[t] / mu0)
        # Scattered after reflecting on the surface
        scatt_surface = (mu0/(mu0-mu[nb_angles])) * (alb_atm/(4*np.pi)) * P0_atm[2*nb_angles-1-nb_angles] * (F0*grd_alb*np.exp(-tauStar_tot/mu0)) * np.exp(-(tauStar_tot-tau[t])/mu0) 
        I1[t, nb_angles] = scatt_before + scatt_direct + scatt_surface
        ## Case µ=µ0
        if np.any(mask_mu0):
            # Alreday scattered
            scatt_before = I1[idx_up, m_arr[mask_mu0]] * np.exp(-(tau[idx_up]-tau[t])/mu[m_arr[mask_mu0]])
            # Scattered before reaching the surface
            scatt_direct = (mu0/(mu0+mu[m_arr[mask_mu0]])) * (alb_atm/(4*np.pi)) * P0_atm[m_arr[mask_mu0]] * F0 * (np.exp(-tau[t] / mu0) - np.exp(-tau[idx_up] / mu0) * np.exp(-(tau[idx_up]-tau[t])/mu[m_arr[mask_mu0]]))
            # Scattered after reflecting on the ground
            scatt_surface = (alb_atm/(4*np.pi)) * P0_atm[2*nb_angles-1-m_arr[mask_mu0]] * (F0*grd_alb*np.exp(-tauStar_tot/mu0)) * np.exp(-(tauStar_tot-tau[t])/mu0) * (tau[idx_up-1]-tau[t]) / mu0
            I1[t, m_arr[mask_mu0]] = scatt_before + scatt_direct + scatt_surface
        




    # Successive orders of scattering n>=2
    In_1 = I1
    I = copy.deepcopy(I1)
    I_saved = [] # saving SOS fields for ploting graphes
    I_saved.append(I1)
    In=np.ones((nb_layers, 2*nb_angles))
    n=1

    while max(max(In[0,nb_angles:]/I[0,nb_angles:]), max(In[nb_layers-1,:nb_angles]/I[nb_layers-1,:nb_angles])) >= 0.0001:
        n+=1
        print(f'Computing radiance fields for n={n}: In/I = {max(max(In[0,nb_angles:]/I[0,nb_angles:]), max(In[nb_layers-1,:nb_angles]/I[nb_layers-1,:nb_angles]))}')
        In = np.zeros((nb_layers, 2*nb_angles))

        # Computation of source function Jn
        Jn = np.zeros((nb_layers, 2*nb_angles))
        for t in range(nb_layers):
            # Create masks for non-zero mu values
            mu_mask_neg = np.abs(mu[:nb_angles]) > 1e-10
            
            if t>=idx_up and t<=idx_down:
                Jn[t,:] = (alb_atm/4) * np.trapz(P_atm[:,::-1]*In_1[t,:], mu, axis=1) * (dtau_atm/(dtau_atm+dtau_aer)) + (alb_aer/4) * np.trapz(P_aer[:,::-1]*In_1[t,:], mu, axis=1) * (dtau_aer/(dtau_atm+dtau_aer))
            else:
                Jn[t,:] = (alb_atm/4) * np.trapz(P_atm[:,::-1]*In_1[t,:], mu, axis=1)
                

        # Compute downward field for upper atmospheric layer
        m_arr = np.arange(nb_angles-1) # [1, ..., nb_angles-2]
        mask_mu0 = np.abs(mu[m_arr] + mu0) < 0.0001 # angles |µ+µ0|<0.0001
        
        for t in range(idx_up):        
            # general case: µ != 0
            for m in range(nb_angles-1):
                if abs(mu[m]) < MU_THRESHOLD:
                    In[t,m] = improved_asymptotic_downward_radiance(Jn[:t+1,m], tau[:t+1], tau[t], mu[m])
                else:
                    # Standard calculation for larger μ values
                    tau_diff = tau[t] - tau[:t+1]
                    exp_term = np.exp(tau_diff / mu[m])
                    integrand = Jn[:t+1,m] * exp_term
                    In[t,m] = -np.trapz(integrand, tau[:t+1]) / mu[m]
            # Improved interpolation for μ → 0 region (unchanged)
            if tau[idx_up-1] <= 0.0625: idx = int(0.005*nb_angles)
            elif tau[idx_up-1] <= 1: idx = int(0.02*nb_angles)
            elif tau[idx_up-1] < 4: idx = int(0.04*nb_angles)
            else: idx = int(0.06*nb_angles)
            for i in range(idx):
                In[t, nb_angles-1-i] = improved_limit_mu_down(In[t, :nb_angles], mu[:nb_angles], nb_angles, idx, i)

        for t in range(idx_up, idx_down+1):        
            # general case: µ != 0
            for m in range(nb_angles-1):
                if abs(mu[m]) < MU_THRESHOLD:
                    In[t,m] = improved_asymptotic_downward_radiance(Jn[idx_up:t+1,m], tau[idx_up:t+1], tau[t], mu[m])
                else:
                    # Standard calculation for larger μ values
                    tau_diff = tau[t] - tau[idx_up-1:t+1]
                    exp_term = np.exp(tau_diff / mu[m])
                    integrand = Jn[idx_up-1:t+1,m] * exp_term
                    In[t,m] = In[idx_up-1,m] * np.exp((tau[t]-tau[idx_up-1])/mu[m]) - np.trapz(integrand, tau[idx_up-1:t+1]) / mu[m]
            # Improved interpolation for μ → 0 region (unchanged)
            if tau[idx_down] <= 0.0625: idx = int(0.005*nb_angles)
            elif tau[idx_down] <= 1: idx = int(0.02*nb_angles)
            elif tau[idx_down] < 4: idx = int(0.04*nb_angles)
            else: idx = int(0.06*nb_angles)
            for i in range(idx):
                In[t, nb_angles-1-i] = improved_limit_mu_down(In[t, :nb_angles], mu[:nb_angles], nb_angles, idx, i)
        
        for t in range(idx_down+1, nb_layers):        
            # general case: µ != 0
            for m in range(nb_angles-1):
                if abs(mu[m]) < MU_THRESHOLD:
                    In[t,m] = improved_asymptotic_downward_radiance(Jn[idx_down+1:t+1,m], tau[idx_down+1:t+1], tau[t], mu[m])
                else:
                    # Standard calculation for larger μ values
                    tau_diff = tau[t] - tau[idx_down:t+1]
                    exp_term = np.exp(tau_diff / mu[m])
                    integrand = Jn[idx_down:t+1,m] * exp_term
                    In[t,m] = In[idx_down,m] * np.exp((tau[t]-tau[idx_down])/mu[m]) - np.trapz(integrand, tau[idx_down:t+1]) / mu[m]
            # Improved interpolation for μ → 0 region (unchanged)
            if tau[idx_down] <= 0.0625: idx = int(0.005*nb_angles)
            elif tau[idx_down] <= 1: idx = int(0.02*nb_angles)
            elif tau[idx_down] < 4: idx = int(0.04*nb_angles)
            else: idx = int(0.06*nb_angles)
            for i in range(idx):
                In[t, nb_angles-1-i] = improved_limit_mu_down(In[t, :nb_angles], mu[:nb_angles], nb_angles, idx, i)

        
        # Compute upward field for upper atmospheric layer
        m_arr = np.arange(nb_angles+1, 2*nb_angles)

        for t in range(idx_down+1, nb_layers):
            # Vectorized for µ>0 (excluding µ=0)
            tau_diff = tau[t:] - tau[t]
            exp_term = np.exp(-tau_diff[:, None] / mu[m_arr]) 
            # For very optically thick cases, use a different normalization
            if tau[nb_layers-1]/mu[nb_angles+1] >= 50:
                In[t, m_arr] = grd_alb * In[nb_layers-1, nb_angles-1-(m_arr-nb_angles)] * np.exp(-(tau[nb_layers-1]-tau[t])/mu[m_arr]) + np.trapz(Jn[t:, m_arr] * (exp_term / mu[m_arr]), tau[t:], axis=0)
            else:
                In[t, m_arr] = grd_alb * In[nb_layers-1, nb_angles-1-(m_arr-nb_angles)] * np.exp(-(tau[nb_layers-1]-tau[t])/mu[m_arr]) + np.trapz(Jn[t:, m_arr] * exp_term, tau[t:], axis=0) / mu[m_arr]
            # Special case for µ=0
            In[t, nb_angles] = Jn[t, nb_angles]
            # Interpolation for μ → 0 region (unchanged)
            idx = nb_angles+1
            while np.abs((In[t,idx]-In[t,idx+1])-(In[t,idx+1]-In[t,idx+2]))>0.0001:
                idx+=1
            idx+=1
            for m in range(nb_angles+1,idx):
                weight = mu[m]/mu[idx]
                In[t,m]= (1-weight)*In[t, nb_angles] + weight*In[t,idx]
        
        for t in range(idx_up, idx_down+1):
            # Vectorized for µ>0 (excluding µ=0)
            tau_diff = tau[t:idx_down+1] - tau[t]
            exp_term = np.exp(-tau_diff[:, None] / mu[m_arr]) 
            # For very optically thick cases, use a different normalization
            if tau[nb_layers-1]/mu[nb_angles+1] >= 50:
                In[t, m_arr] = In[idx_down+1, m_arr] * np.exp(-(tau[idx_down+1]-tau[t])/mu[m_arr]) + np.trapz(Jn[t:idx_down+1, m_arr] * (exp_term / mu[m_arr]), tau[t:idx_down+1], axis=0)
            else:
                In[t, m_arr] = In[idx_down+1, m_arr] * np.exp(-(tau[idx_down+1]-tau[t])/mu[m_arr]) + np.trapz(Jn[t:idx_down+1, m_arr] * exp_term, tau[t:idx_down+1], axis=0) / mu[m_arr]
            # Special case for µ=0
            In[t, nb_angles] = Jn[t, nb_angles]
            # Interpolation for μ → 0 region (unchanged)
            idx = nb_angles+1
            while np.abs((In[t,idx]-In[t,idx+1])-(In[t,idx+1]-In[t,idx+2]))>0.0001:
                idx+=1
            idx+=1
            for m in range(nb_angles+1,idx):
                weight = mu[m]/mu[idx]
                In[t,m]= (1-weight)*In[t, nb_angles] + weight*In[t,idx]
        
        for t in range(idx_up):
            # Vectorized for µ>0 (excluding µ=0)
            tau_diff = tau[t:idx_up] - tau[t]
            exp_term = np.exp(-tau_diff[:, None] / mu[m_arr]) 
            # For very optically thick cases, use a different normalization
            if tau[nb_layers-1]/mu[nb_angles+1] >= 50:
                In[t, m_arr] = In[idx_up, m_arr] * np.exp(-(tau[idx_up]-tau[t])/mu[m_arr]) + np.trapz(Jn[t:idx_up, m_arr] * (exp_term / mu[m_arr]), tau[t:idx_up], axis=0)
            else:
                In[t, m_arr] = In[idx_up, m_arr] * np.exp(-(tau[idx_up]-tau[t])/mu[m_arr]) + np.trapz(Jn[t:idx_up, m_arr] * exp_term, tau[t:idx_up], axis=0) / mu[m_arr]
            # Special case for µ=0
            In[t, nb_angles] = Jn[t, nb_angles]
            # Interpolation for μ → 0 region (unchanged)
            idx = nb_angles+1
            while np.abs((In[t,idx]-In[t,idx+1])-(In[t,idx+1]-In[t,idx+2]))>0.0001:
                idx+=1
            idx+=1
            for m in range(nb_angles+1,idx):
                weight = mu[m]/mu[idx]
                In[t,m]= (1-weight)*In[t, nb_angles] + weight*In[t,idx]

        
        In_1 = In # recurrence

        for m in range(2*nb_angles):
            for t in range(nb_layers):
                I[t,m] += In[t,m]
        
        I_saved.append(In) # updating the saving list for ploting graphes

    
    
    ## Computing fluxes
    flux_up = np.zeros(nb_layers)
    flux_down = np.zeros(nb_layers)
    for i in range(nb_layers):
        flux_down[i] = np.trapz(I[i, :nb_angles] * mu[:nb_angles], mu[:nb_angles]) - (F0/(4*np.pi))*np.exp(-tau[i]/mu0)
        flux_up[i] = np.trapz(I[i, nb_angles:] * mu[nb_angles:], mu[nb_angles:]) + (F0/(4*np.pi))*grd_alb*np.exp(-(2*tau[nb_layers-1]-tau[i])/mu0)
    net_flux_toa = - flux_down[0] - flux_up[0]

    ## Returning result
    if tauStar_aer == 0:
        return net_flux_toa
    else:
        delta_F = net_flux_toa - SOS_Aer_radiative_forcing(0, dtau_aer, tauStar_atm, dtau_atm, P_aer, P0_aer, alb_aer, P_atm, P0_atm, alb_atm, grd_alb, F0, mu, mu0, nb_angles, tau, nb_layers, idx_up, idx_down)
        return delta_F

################################################ 
#           Compute radiative forcing
################################################ 
def SOS_Aer_critical_albedo(tauStar_aer, dtau_aer, tauStar_atm, dtau_atm, P_aer, P0_aer, P_atm, P0_atm, alb_atm, grd_alb, F0, mu, mu0, nb_angles, tau, nb_layers, idx_up, idx_down):
    alb_max = 1
    alb_min = 0
    while (alb_max-alb_min)>0.1:
        # Dichotomous research
        alb_aer_tested = (alb_max+alb_min)/2
        radiative_forcing = SOS_Aer_radiative_forcing(tauStar_aer, dtau_aer, tauStar_atm, dtau_atm, P_aer, P0_aer, alb_aer_tested, P_atm, P0_atm, alb_atm, grd_alb, F0, mu, mu0, nb_angles, tau, nb_layers, idx_up, idx_down)
        # Tested albedo is close enough from the critical albedo: return the tested albedo
        if np.abs(radiative_forcing)<0.001:
            return alb_aer_tested
        else:
            if radiative_forcing>0:
                alb_min = alb_aer_tested
            else:
                alb_max = alb_aer_tested

    return (alb_max+alb_min)/2


###########################################################
#                   MAIN SCRIPT
###########################################################

## Phase function considered
aer_phase_fun = ''

## Working folder
folder = fr'D:\Polytechnique\4_3A\STAGE_3A\Harvard\SOS DYKEMA\Code perso\SOS_AER\Critical_Albedo_Map'
os.makedirs(folder, exist_ok=True)
aer_phase_function_filename = f'Phase_function_X.txt'
result_file_name = f'Results_function_X.txt'
full_path_phase_function = os.path.join(folder, aer_phase_function_filename)
full_path_result = os.path.join(folder, result_file_name)

## Angles
# Solar angular position (mu0=cos(theta_solar))
mu0 = 0.5
# Defining angles list
nb_angles = 501 # number of points for phase function definition
mu_neg = np.linspace(-1, 0, nb_angles) # [-1, ... 0]
mu_pos = np.linspace(0, 1, nb_angles) # [0, ... 1]
mu = np.concatenate((mu_neg, mu_pos)) # [-1, ... 0, 0, ... 1]
µ_1, µ_2 = mu_approx_In(mu, nb_angles) # defining µ_1 and µ_2 for transition layer in In approx (µ -> 0)

## Atmospheric molecules properties 
# Defining atmospheric phase function
atm_phase_fun = 'rayleigh'  # 'iso', 'hg', 'fwc', 'mie', 'rayleigh' or 'eva'
# Single scattering albedo
alb_atm = 1
# Henyey-Greenstein parameters
g_atm = 0.5
# Mie parameters
r_atm = 0 # radius (in meter)
lambda0_atm = 0 # wavelength (in meter)
indx_atm = 0 # complex refractive index
N0_atm = None
r_m_atm = None
sig_atm = None
P0_atm, P_atm = phase_func('atm', atm_phase_fun, g_atm, r_atm, lambda0_atm, indx_atm, nb_angles, mu, mu0, N0_atm, r_m_atm, sig_atm)

    
## Aerosol layer phase function
aer_phase_fun = 'eva'  # 'iso', 'hg', 'fwc', 'mie', 'rayleigh', 'wildfire' 'eva'   
# Henyey-Greenstein parameters
g_aer = 0.5
# Mie parameters
r_aer = 0 # radius (in µm)
lambda0_aer = 0.550 # wavelength (in µm) -> miepython uses wl in µm
indx_aer = 1.44 + 0.0j # complex refractive index
N0_aer = 501187 # log-normal distrib density (in cm^-3)
r_m_aer = 0.506 # log-normal distrib mode radius (in µm)
sig_aer = 1.2 # log-normal distrib standard deviation
P0_aer, P_aer = phase_func('aer', aer_phase_fun, g_aer, r_aer, lambda0_aer, indx_aer, nb_angles, mu, mu0, N0_aer, r_m_aer, sig_aer)

## Flux
F0 = np.pi/mu0 # normalized over azimuthal angle Phi (according to VdH convention)

## Ground reflectivity/albedo
grd_alb = 0.15

## Optical depth profile
# Altitudes
z0 = 120 # Atmosphere altitude (in km)
z_up = 25 # Upper boundary of aerosol layer (in km)
z_down = 17 # Lower boundary of aerosol layer (in km)
if z_down > z_up: z_down, z_up = z_up, z_down # sort altitudes if needed
# Defining optical depth profile
nb_layers = 800
tauStar_atm = 0.104 # Initial atmopshere optical depth without aerosols
tauStar_aer_list = [0.120] # Aerosols optical depth
result = np.zeros(len(tauStar_aer_list))
for idx, tauStar_aer in enumerate(tauStar_aer_list):
    tauStar_tot = tauStar_atm + tauStar_aer
    tau = tau_profile(tauStar_atm, tauStar_aer, z0, z_up, z_down, nb_layers) # overall optical depth profile
    z_profile = np.linspace(z0, 0, nb_layers)
    idx_up, idx_down = np.argmin(np.abs(z_profile - z_up)), np.argmin(np.abs(z_profile - z_down))
    # Optical depth step in the layer
    dtau_aer = tauStar_aer / (idx_down+1-idx_up)
    dtau_atm = tauStar_atm / nb_layers

    # run simulation
    result[idx] = SOS_Aer_critical_albedo(tauStar_aer, dtau_aer, tauStar_atm, dtau_atm, P_aer, P0_aer, P_atm, P0_atm, alb_atm, grd_alb, F0, mu, mu0, nb_angles, tau, nb_layers, idx_up, idx_down)

# Use multiprocessing to calculate in parallel
with open(full_path_result, 'w', encoding='utf-8') as f:
    f.write(f'Phase function = {aer_phase_fun}')
    f.write(f'mu0=0.5\n')
    f.write('tauStar_aer \t critic_alb\n')
    for idx in range(len(tauStar_aer_list)):
        f.write(f"{tauStar_aer_list[idx]}\t{result[idx]}\n")