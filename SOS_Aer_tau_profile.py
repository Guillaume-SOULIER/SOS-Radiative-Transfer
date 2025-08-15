import numpy as np
import os
import matplotlib.pyplot as plt

def tau_profile(tauStar_atm, tauStar_aer, z0, z_up, z_down, nb_layers):
    """
    Compute the cumulative optical depth profile for an atmosphere with:
    - atmospheric molecules (0 to z0, total tauStar_atm)
    - an aerosol layer (z1 to z2, total tauStar_aer)
    Returns:
        tau_profile: array of length nb_layers, cumulative optical depth from top (0) to bottom (tauStar_atm+tauStar_aer)
    """
    print('Computing Optical depth profile')

    # Altitude grid (0 = top, z0 = bottom)
    z_profile = np.linspace(z0, 0, nb_layers)
    
    idx_up, idx_down = np.argmin(np.abs(z_profile - z_up)), np.argmin(np.abs(z_profile - z_down))

    # Uniform molecular extinction over [0, z0]
    tau = np.arange(0, nb_layers) * tauStar_atm / (nb_layers-1)  # extinction per unit length

    # Aerosol extinction only between z1 and z2
    dtau_aer = tauStar_aer / (idx_down+1-idx_up)
    for i in range(idx_up, nb_layers):
        if i<= idx_down: tau[i] += (i+1-idx_up) * dtau_aer
        else: tau[i] += tauStar_aer
    
    plt.plot(tau, z_profile)
    plt.xlabel('Cumulative optical depth')
    plt.ylabel('Altitude (km)')

    plt.title('Optical depth profile (atm molecules and aerosols)') # (molecules + aerosols)')
    # plt.gca().invert_yaxis()
    plt.grid(True)
    plt.show()

    save_fig = False
    if save_fig :
        graphe_folder = fr'D:\Polytechnique\4_3A\STAGE_3A\Harvard\SOS DYKEMA\Code perso\SOS_AER'
        os.makedirs(graphe_folder, exist_ok=True)
        filename_png = f'Tau_profile_atm_mol_and_aer_volcanic.png'
        path = os.path.join(graphe_folder, filename_png)
        plt.savefig(path, dpi=600)
        print(f'Graphe saved in {path}')
        plt.close()

    
    



    return tau



## Testing the function

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    # Example parameters
    tauStar_atm = 0.2
    tauStar_aer = 0.1
    z0 = 10000  # 10 km
    z1 = 4000   # 4 km
    z2 = 2000   # 2 km
    nb_layers = 500

    tau_prof = tau_profile(tauStar_atm, tauStar_aer, z0, z1, z2, nb_layers)
    z_profile = np.linspace(0, z0, nb_layers)

    plt.plot(tau_prof, z_profile)
    plt.xlabel('Cumulative optical depth')
    plt.ylabel('Altitude (m)')

    plt.title('Optical depth profile (molecules + aerosols)')
    plt.gca().invert_yaxis()
    plt.grid(True)
    plt.show()
