# SOS-Radiative-Transfer
Code developed in partnership with Harvard (with support from NASA) to simulate the path of light in the atmosphere with an aerosol layer (forest fires, volcanic eruptions) using the SOS method.

## Overview
**SOS_AER** is a Python code that simulates radiative transfer in a plane-parallel atmosphere containing an aerosol layer, using the **Successive Orders of Scattering (SOS)** method.

The atmosphere is modeled as a stack of discrete layers, with a mixture of molecules and aerosols whose optical properties and phase functions can vary.  
The code computes:
- Upward and downward radiance fields for each scattering order.
- Integrated fluxes.
- Mean diffusivity.
- Radiative heating rate profiles.

**Angles µ** follow the convention:
- `µ > 0`: upward directions (outgoing to space).
- `µ < 0`: downward directions (toward the surface).

## Main Features
- **Boundary conditions**:
  - Surface with *Lambertian* or *specular* reflectivity (user-defined albedo).
  - Top of atmosphere: direct solar illumination with given solar zenith (`µ0`).
- **Supported phase functions**:
  - Isotropic
  - Henyey-Greenstein
  - Rayleigh
  - Mie (monodisperse)
  - FWC (*Full Width Cloud* tabulated data)
  - Log-normal Mie for specific clouds or aerosols (`eva`, `wildfire`, …)
- **Molecule/aerosol mixture**:
  - Optical properties and phase functions computed as a weighted mix by optical thickness.
- **Numerical optimizations**:
  - Accurate handling of `µ → 0` limits (stable interpolation and asymptotic expansions).
  - Vectorized NumPy computations.
  - Phase function caching to avoid recomputation.
- **Graphical outputs**:
  - Flux profiles
  - Diffusivity profiles
  - Heating rate profiles
  - Diffusivity profiles by scattering order

## File Structure

| File | Role |
|------|------|
| **SOS_Aer_main.py** | Main program: sets parameters, calls computational routines, and generates outputs. |
| **SOS_Aer_tau_profile.py** | Builds the cumulative optical depth profile for the atmosphere and aerosol layer. |
| **SOS_Aer_phase_func.py** | Computes phase functions according to the selected model (isotropic, HG, Mie, Rayleigh, FWC, log-normal). |
| **SOS_Aer_global_va.py** | Global variables, thresholds for small-µ handling, storage/loading of phase functions. |
| **SOS_Aer_fwc_data.py** | Tabulated data for the FWC phase function. |
| **I1_In.py** | Functions to compute radiance fields: first order (`I1_NumInt`), higher orders (`Jn_NumInt`, `In_NumInt`), and µ → 0 approximations. |
| **SOS_Aer_In_limit.py** | Improved asymptotic methods and stable interpolation for near-zero µ. |
| **SOS_Aer_vdh_extract.py** | Utility functions for extracting subsets of angles and radiances in Van de Hulst format. |
| **SOS_Aer_graphe.py** | Plotting functions for flux, diffusivity, and heating rate profiles. |

## Installation & Dependencies
Compatible with **Python 3.8+**. Install dependencies with:
```bash
pip install numpy matplotlib tqdm miepython
```

---

## Usage

### 1. Configure parameters
In `SOS_Aer_main.py`, adjust:
- **Solar parameters**: `mu0` (cosine of solar zenith angle)
- **Altitudes of the aerosol layer**: `z0`, `z_up`, `z_down` (in km)
- **Optical depths**: `tauStar_atm`, `tauStar_aer`
- **Surface albedo**: `grd_alb`
- **Phase functions**:
  - Atmosphere (`atm_phase_fun`, associated parameters)
  - Aerosols (`aer_phase_fun`, associated parameters)

### 2. Run the simulation
```bash
python SOS_Aer_main.py
```

### 3. Outputs
- Console output: progress of scattering order computations.
- Interactive plots: optical depth, flux, diffusivity, heating rate profiles.
- PNG figure saving if enabled in `SOS_Aer_graphe.py`.
- Phase functions stored as `.npy` and `.txt` for reuse.

---

## Example scenario: Easy Volanic Aerosols (EVA) model
- Pure molecular atmosphere + thin absorbing aerosol layer.
- µ0 = 0.5 (~60° solar elevation).
- Rayleigh phase function for molecules, log-normal Mie for aerosols.
- Computation until convergence of `In/I` ratio (< 10⁻⁴ at surface and TOA).

## Example scenario: Easy Volanic Aerosols (EVA) model
- Pure molecular atmosphere + thin absorbing aerosol layer.
- µ0 = 0.5 (~60° solar elevation).
- Rayleigh phase function for molecules, log-normal Mie for aerosols.
- Computation until convergence of `In/I` ratio (< 10⁻⁴ at surface and TOA).

---

## References
- Van de Hulst, H.C. *Multiple Light Scattering*, Vol. 1 & 2.
- Chandrasekhar, S. *Radiative Transfer*.
- Lenoble, J. *Radiative Transfer in Scattering and Absorbing Atmospheres*.

