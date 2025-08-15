import numpy as np


# Configuration parameters for μ → 0 handling
MU_THRESHOLD = 0.01  # Threshold for switching to asymptotic methods
MU_EXTREME_THRESHOLD = 1e-8  # Threshold for extremely small μ
MU_VERY_SMALL_THRESHOLD = 0.001  # Threshold for very small μ


# Atmospheric variables
last_P_atm = None
last_P0_atm = None
last_mu0_atm = None
last_g_atm = None
last_atm_phase_func = None

def save_atm_parameters(last_mu0_atm, last_g_atm, last_atm_phase_func):
    np.save('last_mu0_atm.npy', last_mu0_atm)
    np.save('last_g_atm.npy', last_g_atm)
    np.save('last_atm_phase_func.npy', last_atm_phase_func)

def load_atm_parameters():
    global last_mu0_atm, last_g_atm, last_atm_phase_func
    try:
        last_mu0_atm = np.load('last_mu0_atm.npy')
        last_g_atm = np.load('last_g_atm.npy')
        last_atm_phase_func = np.load('last_atm_phase_func.npy')
        return last_mu0_atm, last_g_atm, last_atm_phase_func
    except FileNotFoundError:
        return None, None, None

def save_atm_phase_functions(last_P_atm, last_P0_atm):
    if last_P_atm is not None:
        np.save('last_P_atm.npy', last_P_atm)
    if last_P0_atm is not None:
        np.save('last_P0_atm.npy', last_P0_atm)

def load_atm_phase_functions():
    global last_P_atm, last_P0_atm
    try:
        last_P_atm = np.load('last_P_atm.npy')
        last_P0_atm = np.load('last_P0_atm.npy')
        return last_P0_atm, last_P_atm
    except FileNotFoundError:
        return None, None


# Aerosols variables
last_P_aer = None
last_P0_aer = None
last_mu0_aer = None
last_g_aer = None
last_aer_phase_func = None

def save_aer_parameters(last_mu0_aer, last_g_aer, last_aer_phase_func):
    np.save('last_mu0_aer.npy', last_mu0_aer)
    np.save('last_g_aer.npy', last_g_aer)
    np.save('last_aer_phase_func.npy', last_aer_phase_func)

def load_aer_parameters():
    global last_mu0_aer, last_g_aer, last_aer_phase_func
    try:
        last_mu0_aer = np.load('last_mu0_aer.npy')
        last_g_aer = np.load('last_g_aer.npy')
        last_aer_phase_func = np.load('last_aer_phase_func.npy')
        return last_mu0_aer, last_g_aer, last_aer_phase_func
    except FileNotFoundError:
        return None, None, None

def save_aer_phase_functions(last_P_aer, last_P0_aer):
    if last_P_aer is not None:
        np.save('last_P_aer.npy', last_P_aer)
    if last_P0_aer is not None:
        np.save('last_P0_aer.npy', last_P0_aer)

def load_aer_phase_functions():
    global last_P_aer, last_P0_aer
    try:
        last_P_aer = np.load('last_P_aer.npy')
        last_P0_aer = np.load('last_P0_aer.npy')
        return last_P0_aer, last_P_aer
    except FileNotFoundError:
        return None, None


# Charge automatiquement au démarrage
load_atm_phase_functions()
load_aer_phase_functions()