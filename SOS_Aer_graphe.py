import numpy as np
import os
import matplotlib.pyplot as plt

# Secondary functions for graphe plots
def graphe_diffusivity(I, mu, z_profile, nb_layers, aer_phase_fun):

    dif = np.zeros(nb_layers)
    for i in range(nb_layers):
        dif[i] = - np.trapz(I[i,:] * mu, mu) / np.trapz(I[i,:], mu)

    plt.plot(dif, z_profile)
    plt.xlabel(rf'Diffusivity $\bar{{\mu}}$')
    plt.ylabel('Altitude (km)')

    plt.title(rf'Diffusivity $\bar{{\mu}}$ for {aer_phase_fun} layer') # (molecules + aerosols)')
    # plt.gca().invert_yaxis()
    plt.grid(True)
    plt.show()

    save_fig = True
    if save_fig :
        graphe_folder = fr'D:\Polytechnique\4_3A\STAGE_3A\Harvard\SOS DYKEMA\Code perso\SOS_AER'
        os.makedirs(graphe_folder, exist_ok=True)
        filename_png = f'Diffusivity_profile_atm_mol_and_aer_{aer_phase_fun}.png'
        path = os.path.join(graphe_folder, filename_png)
        plt.savefig(path, dpi=600)
        print(f'Graphe saved in {path}')
        plt.close()

    
    




def graphe_flux(I, mu, z_profile, nb_layers, nb_angles, tau, mu0, F0, grd_alb, aer_phase_fun):
    
    flux = np.zeros(nb_layers)
    for i in range(nb_layers):
        flux[i] = np.trapz(I[i, :] * mu, mu) - F0*np.exp(-tau[i]/mu0) + grd_alb*F0*np.exp(-(2*tau[nb_layers-1]-tau[i])/mu0)

    plt.plot(flux, z_profile)
    plt.xlabel(rf'Flux')
    plt.ylabel('Altitude (km)')

    plt.title(f'Flux for {aer_phase_fun} layer') # (molecules + aerosols)')
    # plt.gca().invert_yaxis()
    plt.grid(True)
    plt.show()

    save_fig = False
    if save_fig :
        graphe_folder = fr'D:\Polytechnique\4_3A\STAGE_3A\Harvard\SOS DYKEMA\Code perso\SOS_AER'
        os.makedirs(graphe_folder, exist_ok=True)
        filename_png = f'Flux_profile_atm_mol_and_aer_{aer_phase_fun}.png'
        path = os.path.join(graphe_folder, filename_png)
        plt.savefig(path, dpi=600)
        print(f'Graphe saved in {path}')
        plt.close()

    
    




def graphe_heating_rate(I, mu, z_profile, nb_layers, nb_angles, idx_up, idx_down, F0, mu0, tau, grd_alb, aer_phase_fun):
    
    # Physical parameters
    rho = 1.225 # density (in m^-3)
    c_p = 1004 # specific heat (in J.kg^-1.K-1)

    flux_up = np.zeros(nb_layers)
    flux_down = np.zeros(nb_layers)
    for i in range(nb_layers):
        flux_down[i] = np.trapz(I[i, :nb_angles] * mu[:nb_angles], mu[:nb_angles]) - (F0/(4*np.pi))*np.exp(-tau[i]/mu0)
        flux_up[i] = np.trapz(I[i, nb_angles:] * mu[nb_angles:], mu[nb_angles:]) + (F0/(4*np.pi))*grd_alb*np.exp(-(2*tau[nb_layers-1]-tau[i])/mu0)
    
    flux = (flux_down + flux_up)

    heating_rate = np.zeros(nb_layers)
    for i in range(nb_layers-1):
        heating_rate[i] = - (1/(rho*c_p)) * (flux[i+1]-flux[i]) / (z_profile[i+1]-z_profile[i])
    heating_rate[-1] = heating_rate[-2]

    # Erase pics on the boundaries of aerosols layer
    erase_pics = True
    if erase_pics:
        heating_rate[idx_up-1]=heating_rate[idx_up-2]
        heating_rate[idx_down] = heating_rate[idx_down-1]

    
    plt.plot(heating_rate, z_profile)
    plt.xlabel(rf'Heating rate')
    # plt.xscale('log')
    plt.ylabel('Altitude (km)')

    plt.title(f'Heating rate for {aer_phase_fun} layer') # (molecules + aerosols)')
    # plt.gca().invert_yaxis()
    plt.grid(True)
    plt.show()

    save_fig = True
    if save_fig :
        graphe_folder = fr'D:\Polytechnique\4_3A\STAGE_3A\Harvard\SOS DYKEMA\Code perso\SOS_AER'
        os.makedirs(graphe_folder, exist_ok=True)
        filename_png = f'Heating rate_profile_atm_mol_and_aer_{aer_phase_fun}.png'
        path = os.path.join(graphe_folder, filename_png)
        plt.savefig(path, dpi=600)
        print(f'Graphe saved in {path}')
        plt.close()

    
    


def graphe_successive_dif(I_saved, mu, z_profile, nb_layers, nb_angles, aer_phase_fun):

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

    plt.title(rf'Diffusivity $\bar{{\mu}}$ for {aer_phase_fun} layer (SOS)') # (molecules + aerosols)')
    # plt.gca().invert_yaxis()
    plt.grid(True)
    plt.legend()
    plt.show()

    save_fig = True
    if save_fig :
        graphe_folder = fr'D:\Polytechnique\4_3A\STAGE_3A\Harvard\SOS DYKEMA\Code perso\SOS_AER'
        os.makedirs(graphe_folder, exist_ok=True)
        filename_png = f'Diffusivity_profile_atm_mol_and_aer_{aer_phase_fun}_SOS.png'
        path = os.path.join(graphe_folder, filename_png)
        plt.savefig(path, dpi=600)
        print(f'Graphe saved in {path}')
        plt.close()

    
def graphe_flux_up_down(I, mu, z_profile, nb_layers, nb_angles, tau, mu0, F0, grd_alb, aer_phase_fun):
    
    flux_up = np.zeros(nb_layers)
    flux_down = np.zeros(nb_layers)
    for i in range(nb_layers):
        flux_down[i] = np.trapz(I[i, :nb_angles] * mu[:nb_angles], mu[:nb_angles]) - F0*np.exp(-tau[i]/mu0)
        flux_up[i] = np.trapz(I[i, nb_angles:] * mu[nb_angles:], mu[nb_angles:]) + grd_alb*F0*np.exp(-(2*tau[nb_layers-1]-tau[i])/mu0)
    


    plt.plot(flux_up, z_profile, label='Flux up')
    plt.plot(flux_down, z_profile, label='Flux down')
    plt.xlabel(rf'Flux')
    plt.ylabel('Altitude (km)')

    plt.title(f'Flux for {aer_phase_fun} layer') # (molecules + aerosols)')
    # plt.gca().invert_yaxis()
    plt.grid(True)    
    plt.legend()
    plt.show()

    save_fig = False
    if save_fig :
        graphe_folder = fr'D:\Polytechnique\4_3A\STAGE_3A\Harvard\SOS DYKEMA\Code perso\SOS_AER'
        os.makedirs(graphe_folder, exist_ok=True)
        filename_png = f'Flux_profile_atm_mol_and_aer_{aer_phase_fun}.png'
        path = os.path.join(graphe_folder, filename_png)
        plt.savefig(path, dpi=600)
        print(f'Graphe saved in {path}')
        plt.close()