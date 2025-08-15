import numpy as np
import os
import matplotlib.pyplot as plt

# Secondary functions for graphe plots
def graphe_diffusivity(I, mu, z_profile, nb_layers):

    dif = np.zeros(nb_layers)
    for i in range(nb_layers):
        dif[i] = - np.trapz(I[i,:] * mu, mu) / np.trapz(I[i,:], mu)

    plt.plot(dif, z_profile)
    plt.xlabel(rf'Diffusivity $\bar{{\mu}}$')
    plt.ylabel('Altitude (km)')

    plt.title(rf'Diffusivity $\bar{{\mu}}$ for EVA layer') # (molecules + aerosols)')
    # plt.gca().invert_yaxis()
    plt.grid(True)
    plt.show()

    save_fig = True
    if save_fig :
        graphe_folder = fr'D:\Polytechnique\4_3A\STAGE_3A\Harvard\SOS DYKEMA\Code perso\SOS_AER'
        os.makedirs(graphe_folder, exist_ok=True)
        filename_png = f'Diffusivity_profile_atm_mol_and_aer_volcanic.png'
        path = os.path.join(graphe_folder, filename_png)
        plt.savefig(path, dpi=600)
        print(f'Graphe saved in {path}')
        plt.close()

    
    




def graphe_flux(I, mu, z_profile, nb_layers, nb_angles):
    
    flux = np.zeros(nb_layers)
    for i in range(nb_layers):
        flux[i] = - np.trapz(I[i, :] * mu, mu)

    plt.plot(flux, z_profile)
    plt.xlabel(rf'Flux')
    plt.ylabel('Altitude (km)')

    plt.title('Flux for EVA layer') # (molecules + aerosols)')
    # plt.gca().invert_yaxis()
    plt.grid(True)
    plt.show()

    save_fig = True
    if save_fig :
        graphe_folder = fr'D:\Polytechnique\4_3A\STAGE_3A\Harvard\SOS DYKEMA\Code perso\SOS_AER'
        os.makedirs(graphe_folder, exist_ok=True)
        filename_png = f'Flux_profile_atm_mol_and_aer_volcanic.png'
        path = os.path.join(graphe_folder, filename_png)
        plt.savefig(path, dpi=600)
        print(f'Graphe saved in {path}')
        plt.close()

    
    




def graphe_heating_rate(I, mu, z_profile, nb_layers, nb_angles):
    
    # Physical parameters
    rho = 1.225 # density (in m^-3)
    c_p = 1004 # specific heat (in J.kg^-1.K-1)
    
    mu_pos = mu[:nb_angles]
    mu_neg = mu[nb_angles:]
    I_pos = I[:, :nb_angles]
    I_neg = I[:, nb_angles:]
    
    flux = np.zeros(nb_layers)
    for i in range(nb_layers):
        flux[i] = - np.trapz(I_pos[i, :] * mu_pos, mu_pos) - np.trapz(I_neg[i, :] * mu_neg, mu_neg)

    heating_rate = np.zeros(nb_layers)
    for i in range(nb_layers-1):
        heating_rate[i] = - (1/(rho*c_p)) * (flux[i+1]-flux[i]) / (z_profile[i+1]-z_profile[i])
    heating_rate[-1] = heating_rate[-2]

    
    plt.plot(heating_rate, z_profile)
    plt.xlabel(rf'Heating rate')
    # plt.xscale('log')
    plt.ylabel('Altitude (km)')

    plt.title('Heating rate for EVA layer') # (molecules + aerosols)')
    # plt.gca().invert_yaxis()
    plt.grid(True)
    plt.show()

    save_fig = False
    if save_fig :
        graphe_folder = fr'D:\Polytechnique\4_3A\STAGE_3A\Harvard\SOS DYKEMA\Code perso\SOS_AER'
        os.makedirs(graphe_folder, exist_ok=True)
        filename_png = f'Heating rate_profile_atm_mol_and_aer_volcanic.png'
        path = os.path.join(graphe_folder, filename_png)
        plt.savefig(path, dpi=600)
        print(f'Graphe saved in {path}')
        plt.close()

    
    


def graphe_successive_dif(I_saved, mu, z_profile, nb_layers, nb_angles):

    I = np.zeros(nb_layers)
    max_order = len(I_saved)
    
    for m in range(max_order):
        I = I_saved[m]
        if I.shape != ((nb_layers, 2*nb_angles)): print('Error in I_saved memory, not the good shape')

        dif = np.zeros(nb_layers)
        for i in range(nb_layers):
            dif[i] = - np.trapz(I[i,:] * mu, mu) / np.trapz(I[i,:], mu)
        plt.plot(dif, z_profile, label=f'order={m+1}')

    plt.xlabel(rf'Diffusivity $\bar{{\mu}}$')
    plt.ylabel('Altitude (km)')

    plt.title(rf'Diffusivity $\bar{{\mu}}$ for EVA layer (SOS)') # (molecules + aerosols)')
    # plt.gca().invert_yaxis()
    plt.grid(True)
    plt.legend()
    plt.show()

    save_fig = True
    if save_fig :
        graphe_folder = fr'D:\Polytechnique\4_3A\STAGE_3A\Harvard\SOS DYKEMA\Code perso\SOS_AER'
        os.makedirs(graphe_folder, exist_ok=True)
        filename_png = f'Diffusivity_profile_atm_mol_and_aer_volcanic_SOS.png'
        path = os.path.join(graphe_folder, filename_png)
        plt.savefig(path, dpi=600)
        print(f'Graphe saved in {path}')
        plt.close()

    
    