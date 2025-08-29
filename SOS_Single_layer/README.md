# SOS-Radiative-Transfer

## Overview
The version of the SOS_AER code presented here is the version originally developed by the author. It is a **direct transcription of the SOS equations by Q. Min and M. Duan**, and applies to a plane-parallel atmosphere with an absorbing surface and a and a single type of particle (atmospheric molecule or aerosol). Its use can be very relevant for studying the properties of a single layer (cloud, aerosol layer, diffusing atmosphere).

The code computes:
- Upward and downward radiance fields for each scattering order and for the total series (computed for a default accuracy of 100 ppm).
- A comparison of the results with Van de Hulst's reference table (flux, successive scattering orders of radiance fields).
- An estimated scattering order map to reconstruct a series of radiance fields with an accuracy of 100 ppm.
- A map of scattering as a function of optical thickness and solar incidence.

**Angular convention** follow the convention:
- `µ > 0`: upward directions (outgoing to space).
- `µ < 0`: downward directions (toward the surface).

## Main Features
- **Boundary conditions**:
  - Absorbing surface ($\rho_{grd}=0$).
  - Top of atmosphere (arbitrary altitude): direct solar illumination with given solar zenith (`µ0`).
- **Supported phase functions**:
  - Isotropic
  - Henyey-Greenstein
  - Rayleigh
  - Mie (monodisperse)
  - FWC (*Full Width Cloud* tabulated data)
  - Stratocumulus II
- **Numerical optimizations**:
  - Accurate handling of `µ → 0` limits (stable interpolation and asymptotic expansions).
  - Vectorized NumPy computations.
  - Phase function caching to avoid recomputation.
- **Graphical outputs**:
  - Van de Hulst comparison
  - Diffusivity map
  - Scattering order associated with a 100 ppm accuracy

## File Structure

| File | Role |
|------|------|
| **vdh_hg_comparison.py** | Main program that computes successive orders of scattering of the radiance field, displays the corresponding graphs, and compares them with Van de Hulst's results where applicable. |
| **I1_In.py** | Functions to compute radiance fields: first order (`I1_NumInt`), higher orders (`Jn_NumInt`, `In_NumInt`), and µ → 0 approximations. |
| **global_va.py** | Global variables, thresholds for small-µ handling, storage/loading of phase functions. |
| **vdh_extract.py.py** | Returns the values of the fields associated with the viewing angles $µ=0, 0.1, 0.3, 0.5, 0.7, 0.9, 1$ and separates the upward and downward fields. |
| **phase_fun.py** | Computes phase functions according to the selected model (isotropic, HG, Mie, Rayleigh, FWC, Stratocumulus II). |
| **Error_test.py** | Returns the error (in %) associated with comparing the results with those of Van de Hulst (where applicable). |
| **Graphe_N_max.py** | Computes and displays a graph showing the scattering order required to reconstruct the radiance series with an accuracy of 100 ppm. The results are stored in .txt files. |
| **Strato_phase_fun.py** | Computes the Stratocumulus II phase function and saves it to a .txt file using multiprocessing. |
| **vdh_HG.py** | Contains the values from Van de Hulst's data tables, used to compare results in vdh_hg_comparison.py in the case of the Henyey-Greenstein phase function. |
| **vdh_iso.py** | Contains the values from Van de Hulst's data tables, used to compare results in vdh_hg_comparison.py in the case of an isotropic phase function. |


| **SOS_Aer_In_limit.py** | Improved asymptotic methods and stable interpolation for near-zero µ. |
| **SOS_Aer_graphe.py** | Plotting functions for flux, diffusivity, and heating rate profiles. |
| **SOS_Aer_critical albedo.py** | Plotting Haywood critical albedo as a function of optical depth for several phase functions. |

## Installation & Dependencies
Compatible with **Python 3.8+**. Install dependencies with:
```bash
pip install numpy matplotlib tqdm miepython
```

---

## Usage

### 1. Configure parameters
In `SOS_Aer_main_*.py`, adjust:
- **Solar parameters**: `mu0` (cosine of solar zenith angle)
- **Altitudes of the aerosol layer**: `z0`, `z_up`, `z_down` (in km)
- **Optical depths**: `tauStar_atm`, `tauStar_aer`
- **Surface albedo**: `grd_alb`
- **Phase functions**:
  - Atmosphere (`atm_phase_fun`, associated parameters)
  - Aerosols (`aer_phase_fun`, associated parameters)

In `SOS_Aer_tau_profile.py` and `SOS_Aer_graphe.py`, adjust:
- **Name of figures and fodler path**
- **Saving feature** `True` or `False`

In `SOS_Aer_phase_func.py` and `SOS_Aer_graphe.py`, adjust:
- **Name of saving files and fodler path**

### 2. Run the simulation
```bash
python *.py
```

### 3. Outputs
- Console output: progress of scattering order computations.
- Interactive plots: optical depth, flux, diffusivity, heating rate profiles.
- PNG figure saving if enabled in `SOS_Aer_graphe.py`.
- Phase functions stored as `.npy` and `.txt` for faster reuse.

---

## Reference
- M. Duan, Q. Min, 2004, *‘A semi-analytic technique to speed up successive order of scattering model for optically thick media’*, Journal of Quantitative Spectroscopy & Radiative Transfer Vol. 95, p. 21-32
- H. C. Van de Hulst, 1980, ‘Multiple Light Scattering: Tables, Formulas, and Applications. Volume 1’, Academic Press
- H. C. Van de Hulst, 1980, ‘Multiple Light Scattering: Tables, Formulas, and Applications. Volume 2’, Academic Press

---

## Physical formulas for SOS method with single layer and no reflecting surface

## 1st order radiance field

**Downward field** ($\mu \le 0$):  

$$
I_1^{\downarrow}(\tau, \mu \le 0) = I_1^{\downarrow}(\tau_{0},\mu)e^{\tfrac{\tau-\tau_{0}}{\mu}} + \frac{\omega}{4\pi}\frac{\mu_0}{\mu_0+\mu}P(\tau,\mu,\mu_0)F_0\left(e^{-\tfrac{\tau}{\mu_0}}-e^{-\tfrac{\tau_{0}}{\mu_0}}e^{\tfrac{\tau-\tau_{0}}{\mu}}\right)
$$  

**Upward field** ($\mu \ge 0$):  

$$
I_1^{\uparrow}(\tau, \mu \ge 0) = I_1^{\uparrow}(\tau^{\ast},\mu)e^{-\tfrac{\tau^{\ast}-\tau}{\mu}} + \frac{\mu_0}{\mu_0+\mu}\omega\,P(\tau,\mu,\mu_0)\frac{F_0}{4\pi}\left(e^{-\tfrac{\tau}{\mu_0}}-e^{-\tfrac{\tau^{\ast}}{\mu_0}}e^{-\tfrac{\tau^{\ast}-\tau}{\mu}}\right)
$$  

## $n \geq 2$ source function 

**First order**:  

$$
J_1(\tau,\mu) = \frac{\omega}{4\pi}e^{-\tfrac{\tau}{\mu_0}}P(\tau,\mu,\mu_0)F_0
$$  

**For $n \ge 2$**:  

$$
J_n(\tau,\mu) = \frac{\omega}{4}\int_{-1}^{1} P(\tau,\mu,\mu')I_{n-1}(\tau,\mu')d\mu'
$$  

## $n \geq 2$ radiance field  

**Downward field**:  

$$
I_n^{\downarrow}(\tau,\mu \le 0) = I_n^{\downarrow}(\tau_{0},\mu)e^{\tfrac{\tau-\tau_{0}}{\mu}} - \int_{\tau_0}^{\tau} J_n(t,\mu)e^{\tfrac{\tau-t}{\mu}}\frac{dt}{\mu}
$$  

**Upward field**:  

$$
I_n^{\uparrow}(\tau,\mu \ge 0) = I_n^{\uparrow}(\tau^{\ast},\mu)e^{-\tfrac{\tau^{\ast}-\tau}{\mu}} + \int_{\tau}^{\tau^{\ast}} J_n(t,\mu)e^{-\tfrac{t-\tau}{\mu}}\frac{dt}{\mu}
$$  

