# SOS-Radiative-Transfer

## Overview
**SOS_AER** is a Python code that simulates radiative transfer in a plane-parallel atmosphere containing an aerosol layer, using the **Successive Orders of Scattering (SOS)** and the **Discrete Ordinate (DOM** methods.

The atmosphere is modeled as a stack of discrete layers, with a mixture of molecules and aerosols whose optical properties and phase functions can vary.  
The code computes:
- Upward and downward radiance fields for each scattering order and for the total series (computed for a default accuracy of 100 ppm).
- Integrated fluxes (ascending, descending, net).
- Mean diffusivity ($\bar{\mu}\rightarrow 1$: ascending vertical field, $\bar{\mu}\rightarrow -1$: descending vertical field, $\bar{\mu}\rightarrow 0$: horizontal field).
- Radiative heating rate profiles.

**Angular convention** follow the convention:
- `µ > 0`: upward directions (outgoing to space).
- `µ < 0`: downward directions (toward the surface).

## Main Features
- **Boundary conditions**:
  - Surface with *Lambertian* or *specular* reflectivity (user-defined albedo).
  - Top of atmosphere (at $z_atm = 120 km$): direct solar illumination with given solar zenith (`µ0`).
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
| **SOS_Aer_main_lambertian.py** | Main program with a lmabertian surface: sets parameters, calls computational routines, and generates outputs. |
| **SOS_Aer_main_specular.py** | Main program with a specular surface: sets parameters, calls computational routines, and generates outputs. |
| **SOS_Aer_tau_profile.py** | Builds the cumulative optical depth profile for the atmosphere and aerosol layer. |
| **SOS_Aer_phase_func.py** | Computes phase functions according to the selected model (isotropic, HG, Mie, Rayleigh, FWC, log-normal). |
| **SOS_Aer_global_va.py** | Global variables, thresholds for small-µ handling, storage/loading of phase functions. |
| **SOS_Aer_fwc_data.py** | Tabulated data for the FWC phase function. |
| **I1_In.py** | Functions to compute radiance fields: first order (`I1_NumInt`), higher orders (`Jn_NumInt`, `In_NumInt`), and µ → 0 approximations. |
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
python SOS_Aer_main_*.py
```

### 3. Outputs
- Console output: progress of scattering order computations.
- Interactive plots: optical depth, flux, diffusivity, heating rate profiles.
- PNG figure saving if enabled in `SOS_Aer_graphe.py`.
- Phase functions stored as `.npy` and `.txt` for faster reuse.

---

## Example scenario: Easy Volanic Aerosols (EVA) model
- Pure molecular atmosphere + thin absorbing aerosol layer.
- µ0 = 0.5 (~60° solar elevation).
- Layer latitudes: $z_up = 25 km, z_down = 17 km$.
- Rayleigh phase function for molecules ($\omega = 1, \tau_atm^*=0.124$).
- Log-normal Mie for aerosols ($\sigma_v = 1.2, r_m = 0.506 µm, \lambda = 0.550 µm, \omega = 0.97, \tau_aer^* = 0.120, n=1.44$).
- Surface reflectivity: $R_s = 0.15$.
- Computation until convergence of `In/I` ratio (< 10⁻⁴ at surface and TOA).

## Example scenario: Wildfire model
- Pure molecular atmosphere + thin absorbing aerosol layer.
- µ0 = 0.5 (~60° solar elevation).
- Layer latitudes: $z_up = 15 km, z_down = 14 km$.
- Rayleigh phase function for molecules ($\omega = 1, \tau_atm^*=0.124$).
- Log-normal Mie for aerosols ($\sigma_v = 1.5, r_m = 0.065 µm, N = 501,187 cm^-3, \lambda = 0.550 µm, \omega = 0.97, \tau_aer^* = 0.0075, n=1.7 + 0.03j$).
- Surface reflectivity: $R_s = 0.15$.
- Computation until convergence of `In/I` ratio (< 10⁻⁴ at surface and TOA).

---

## References
- M. Duan, Q. Min, 2004, *‘A semi-analytic technique to speed up successive order of scattering model for optically thick media’*, Journal of Quantitative Spectroscopy & Radiative Transfer Vol. 95, p. 21-32
- Toohey, M. Stevens, B. Schmidt, Timmreck, 2016, *‘Easy Volcanic Aerosol (EVA v1.0): an idealized forcing generator for climate simulations’*, Geoscientific Model Development, No. 9, p. 4049-4070
- Y. Li, J. Dykema, 2025, *‘Enhanced Radiative Cooling by Large Aerosol Particles from Pyrocumulonimbus’*, Article under review

---

## Physical formulas for SOS method with Lambertian surface 

## 1st order radiance field  

### Downward field  

**Upper atmospheric layer** ($z_{TOA} \geq z \geq z_{up}$):  

$$
I_1^{\downarrow}(\tau,\mu \leq 0) = \frac{\mu_0}{\mu_0+\mu}\omega_{\text{atm}}P_{\text{atm}}(\mu,\mu_0)\frac{F_0}{4\pi}\left(e^{-\tfrac{\tau}{\mu_0}} - e^{\tfrac{\tau}{\mu}}\right) + \int_0^1 \frac{\mu'}{\mu'-\mu}\omega_{\text{atm}}P_{\text{atm}}(\mu,\mu')\frac{2\rho_{\text{grd}}F_0 e^{-\tfrac{\tau_{\text{atm}}^{\ast}+\tau_{\text{aer}}^{\ast}}{\mu_0}}}{4\pi}\left(e^{-\tfrac{\tau_{\text{atm}}^{\ast}+\tau_{\text{aer}}^{\ast}-\tau}{\mu'}} - e^{-\tfrac{\tau_{\text{atm}}^{\ast}+\tau_{\text{aer}}^{\ast}}{\mu'}}e^{\tfrac{\tau}{\mu}}\right)d\mu'
$$  

**Aerosol layer** ($z_{up} \geq z \geq z_{down}$):  

$$
I_1^{\downarrow}(\tau,\mu \leq 0) = I_1^{\downarrow}(\tau_{up},\mu)e^{\tfrac{\tau-\tau_{up}}{\mu}} + \frac{\mu_0}{\mu_0+\mu}[\frac{d\tau_{\text{atm}}}{d\tau_{\text{atm}}+d\tau_{\text{aer}}}\omega_{\text{atm}}P_{\text{atm}}(\mu,\mu_0)+\frac{d\tau_{\text{aer}}}{d\tau_{\text{atm}}+d\tau_{\text{aer}}}\omega_{\text{aer}}P_{\text{aer}}(\mu,\mu_0)]\frac{F_0}{4\pi}\left(e^{-\tfrac{\tau}{\mu_0}}-e^{-\tfrac{\tau_{up}}{\mu_0}}e^{\tfrac{\tau-\tau_{up}}{\mu}}\right) +
$$

$$
\int_0^1 \frac{\mu'}{\mu'-\mu}[\frac{d\tau_{\text{atm}}}{d\tau_{\text{atm}}+d\tau_{\text{aer}}}\omega_{\text{atm}}P_{\text{atm}}(\mu,-\mu')+\frac{d\tau_{\text{aer}}}{d\tau_{\text{atm}}+d\tau_{\text{aer}}}\omega_{\text{aer}}P_{\text{aer}}(\mu,-\mu')] \frac{2F_0\rho_{\text{grd}}e^{-\tfrac{\tau_{\text{atm}}^{\ast}+\tau_{\text{aer}}^{\ast}}{\mu_0}}}{4\pi}\left(e^{-\tfrac{\tau_{\text{atm}}^{\ast}+\tau_{\text{aer}}^{\ast}-\tau}{\mu'}}-e^{-\tfrac{\tau_{\text{atm}}^{\ast}+\tau_{\text{aer}}^{\ast}-\tau_{up}}{\mu'}}e^{\tfrac{\tau-\tau_{up}}{\mu}}\right)d\mu'
$$  

**Bottom atmospheric layer** ($z_{down} \geq z \geq 0$):  

$$
I_1^{\downarrow}(\tau,\mu \leq 0) = I_1^{\downarrow}(\tau_{down},\mu)e^{\tfrac{\tau-\tau_{down}}{\mu}} + \frac{\mu_0}{\mu_0+\mu}\omega_{\text{atm}}P_{\text{atm}}(\mu,\mu_0)\frac{F_0}{4\pi}\left(e^{-\tfrac{\tau}{\mu_0}}-e^{-\tfrac{\tau_{down}}{\mu_0}}e^{\tfrac{\tau-\tau_{down}}{\mu}}\right) + \int_0^1 \frac{\mu'}{\mu'-\mu}\omega_{\text{atm}}P_{\text{atm}}(\mu,-\mu')\frac{2F_0\rho_{\text{grd}}e^{-\tfrac{\tau_{\text{atm}}^{\ast}+\tau_{\text{aer}}^{\ast}}{\mu_0}}}{4\pi}\left(e^{-\tfrac{\tau_{\text{atm}}^{\ast}+\tau_{\text{aer}}^{\ast}-\tau}{\mu'}}-e^{-\tfrac{\tau_{\text{atm}}^{\ast}+\tau_{\text{aer}}^{\ast}-\tau_{down}}{\mu'}}e^{\tfrac{\tau-\tau_{down}}{\mu}}\right)d\mu'
$$  

### Upward field  

**Bottom atmospheric layer** ($z_{down} \ge z \ge 0$):

$$
I_1^{\uparrow}(\tau,\mu \ge 0) = 2\rho_{\text{grd}}\int_{-1}^0 I_1^{\downarrow}(\tau_{\text{atm}}^{\ast} + \tau_{\text{aer}}^{\ast},\mu')\mu'd\mu'e^{-\tfrac{\tau_{\text{atm}}^{\ast}+\tau_{\text{aer}}^{\ast}-\tau}{\mu}} + \frac{\mu_0}{\mu_0+\mu}\omega_{\text{atm}}P_{\text{atm}}(\mu,\mu_0)\frac{F_0}{4\pi}\left(e^{-\tfrac{\tau}{\mu_0}} - e^{-\tfrac{\tau_{down}}{\mu_0}} e^{-\tfrac{\tau_{down}-\tau}{\mu}}\right) + \int_0^1 \frac{\mu'}{\mu'-\mu}\omega_{\text{atm}}P_{\text{atm}}(\mu,-\mu')\frac{2F_0\rho_{\text{grd}}e^{-\tfrac{\tau_{\text{atm}}^{\ast}+\tau_{\text{aer}}^{\ast}}{\mu_0}}}{4\pi}\left(e^{-\tfrac{\tau_{\text{atm}}^{\ast}+\tau_{\text{aer}}^{\ast}-\tau}{\mu'}} - e^{-\tfrac{\tau_{\text{atm}}^{\ast}+\tau_{\text{aer}}^{\ast}-\tau_{down}}{\mu'}} e^{-\tfrac{\tau_{down}-\tau}{\mu}}\right)d\mu'
$$

**Aerosol layer** ($z_{up} \geq z \geq z_{down}$):  

$$
I_1^{\uparrow}(\tau,\mu \ge 0) = I_1^{\uparrow}(\tau_{\text{down}},\mu)e^{-\tfrac{\tau_{\text{down}}-\tau}{\mu}} + \frac{\mu_0}{\mu_0+\mu}\left(\frac{d\tau_{\text{atm}}}{d\tau_{\text{atm}}+d\tau_{\text{aer}}}\omega_{\text{atm}}P_{\text{atm}}(\mu,\mu_0)+\frac{d\tau_{\text{aer}}}{d\tau_{\text{atm}}+d\tau_{\text{aer}}}\omega_{\text{aer}}P_{\text{aer}}(\mu,\mu_0)\right)\frac{F_0}{4\pi}\left(e^{-\tfrac{\tau}{\mu_0}}-e^{-\tfrac{\tau_{\text{down}}}{\mu_0}}e^{-\tfrac{\tau_{\text{down}}-\tau}{\mu}}\right) + 
$$

$$
\int_0^1 \frac{\mu'}{\mu'-\mu}\left(\frac{d\tau_{\text{atm}}}{d\tau_{\text{atm}}+d\tau_{\text{aer}}}\omega_{\text{atm}}P_{\text{atm}}(\mu,-\mu')+\frac{d\tau_{\text{aer}}}{d\tau_{\text{atm}}+d\tau_{\text{aer}}}\omega_{\text{aer}}P_{\text{aer}}(\mu,-\mu')\right)\frac{2F_0\rho_{\text{grd}}e^{-\tfrac{\tau_{\text{atm}}^{\ast}+\tau_{\text{aer}}^{\ast}}{\mu_0}}}{4\pi}\left(e^{-\tfrac{\tau_{\text{atm}}^{\ast}+\tau_{\text{aer}}^{\ast}-\tau}{\mu'}}-e^{-\tfrac{\tau_{\text{atm}}^{\ast}+\tau_{\text{aer}}^{\ast}-\tau_{\text{down}}}{\mu'}}e^{-\tfrac{\tau_{\text{down}}-\tau}{\mu}}\right)d\mu'
$$  

**Upper atmospheric layer** ($z_{TOA} \geq z \geq z_{up}$):  

$$
I_1^{\uparrow}(\tau,\mu) \ge 0) = I_1^{\uparrow}(\tau_{\text{up}},\mu)e^{-\tfrac{\tau_{up}-\tau}{\mu}} + \frac{\mu_0}{\mu_0+\mu}\omega_{\text{atm}}P_{\text{atm}}(\mu,\mu_0)\frac{F_0}{4\pi}\left(e^{-\tfrac{\tau}{\mu_0}}-e^{-\tfrac{\tau_{up}}{\mu_0}}e^{-\tfrac{\tau_{up}-\tau}{\mu}}\right) + \int_0^1 \frac{\mu'}{\mu'-\mu}\omega_{\text{atm}}P_{\text{atm}}(\mu,-\mu')\frac{2F_0\rho_{\text{grd}}e^{-\tfrac{\tau_{\text{atm}}^{\ast}+\tau_{\text{aer}}^{\ast}}{\mu_0}}}{4\pi}\left(e^{-\tfrac{\tau_{\text{atm}}^{\ast}+\tau_{\text{aer}}^{\ast}-\tau}{\mu'}}-e^{-\tfrac{\tau_{\text{atm}}^{\ast}+\tau_{\text{aer}}^{\ast}-\tau_{up}}{\mu'}}e^{-\tfrac{\tau_{up}-\tau}{\mu}}\right)d\mu'
$$

## $n \geq 2$ source function  

**Upper and bottom atmospheric layers** ($z_{down} \ge z \ge 0$ and $z_{TOA} \geq z \geq z_{up}$):  

$$
J_n(\tau,\mu) = \frac{\omega_{\text{atm}}}{4}\int_{-1}^1 P_{\text{atm}}(\mu,\mu')\,I_{n-1}^{(\uparrow\downarrow)}(\tau,\mu')d\mu'
$$  

**Aerosol layer** ($z_{up} \geq z \geq z_{down}$):  

$$
J_n(\tau,\mu) = \frac{d\tau_{\text{atm}}}{d\tau_{\text{atm}}+d\tau_{\text{aer}}}\frac{\omega_{\text{atm}}}{4}\int_{-1}^1 P_{\text{atm}}(\mu,\mu')I_{n-1}^{(\uparrow\downarrow)}(\tau,\mu')d\mu' + \frac{d\tau_{\text{aer}}}{d\tau_{\text{atm}}+d\tau_{\text{aer}}}\frac{\omega_{\text{aer}}}{4}\int_{-1}^1 P_{\text{aer}}(\mu,\mu')I_{n-1}^{(\uparrow\downarrow)}(\tau,\mu')d\mu'
$$  

## $n \geq 2$ radiance field  

### Downward field  

**Upper atmospheric layer** ($z_{TOA} \geq z \geq z_{up}$):  

$$
I_n^{\downarrow}(\tau,\mu \leq 0) = -\int_0^{\tau} J_n(t,\mu)e^{\tfrac{\tau-t}{\mu}}\frac{dt}{\mu}
$$  

**Aerosol layer** ($z_{up} \geq z \geq z_{down}$):  

$$
I_n^{\downarrow}(\tau,\mu \leq 0) = I_n^{\downarrow}(\tau_{up},\mu)e^{\tfrac{\tau-\tau_{up}}{\mu}} - \int_{\tau_{up}}^{\tau} J_n(t,\mu)e^{\tfrac{\tau-t}{\mu}}\frac{dt}{\mu}
$$  

**Bottom atmospheric layer** ($z_{down} \ge z \ge 0$):  

$$
I_n^{\downarrow}(\tau,\mu \leq 0) = I_n^{\downarrow}(\tau_{down},\mu)e^{\tfrac{\tau-\tau_{down}}{\mu}} - \int_{\tau_{down}}^{\tau} J_n(t,\mu)e^{\tfrac{\tau-t}{\mu}}\frac{dt}{\mu}
$$  

### Upward field  

**Bottom atmospheric layer** ($z_{down} \ge z \ge 0$):  

$$
I_n^{\uparrow}(\tau,\mu \geq 0) = -2\rho_{\text{grd}}\int_{-1}^0 I_n^{\downarrow}(\tau_{\text{atm}}^{\ast}+\tau_{\text{aer}}^{\ast},\mu')\mu'd\mu'e^{-\tfrac{\tau_{\text{atm}}^{\ast}+\tau_{\text{aer}}^{\ast}-\tau}{\mu}} + \int_{\tau}^{\tau_{\text{atm}}^{\ast}+\tau_{\text{aer}}^{\ast}} J_n(t,\mu)e^{-\tfrac{t-\tau}{\mu}}\frac{dt}{\mu}
$$  

**Aerosol layer** ($z_{up} \geq z \geq z_{down}$):  

$$
I_n^{\uparrow}(\tau,\mu \geq 0) = I_n^{\uparrow}(\tau_{down},\mu)e^{-\tfrac{\tau_{down}-\tau}{\mu}} + \int_{\tau}^{\tau_{down}} J_n(t,\mu)e^{-\tfrac{t-\tau}{\mu}}\frac{dt}{\mu}
$$  

**Upper atmospheric layer** ($z_{TOA} \geq z \geq z_{up}$):  

$$
I_n^{\uparrow}(\tau,\mu \geq 0) = I_n^{\uparrow}(\tau_{up},\mu)e^{-\tfrac{\tau_{up}-\tau}{\mu}} + \int_{\tau}^{\tau_{up}} J_n(t,\mu)e^{-\tfrac{t-\tau}{\mu}}\frac{dt}{\mu}
$$  

## Total flux  

$$
F^{\uparrow}(\tau) = \sum_{k=1}^{\infty}\left[\int_0^1 I_k^{\uparrow}(\tau,\mu)\mu d\mu\right] + \int_0^1 \mu\left[2\rho_{\text{grd}}F_0 e^{-\tfrac{\tau_{\text{atm}}^{\ast}+\tau_{\text{aer}}^{\ast}}{\mu_0}}\right]e^{-\tfrac{\tau_{\text{atm}}^{\ast}+\tau_{\text{aer}}^{\ast}-\tau}{\mu}}d\mu
$$  

$$
F^{\downarrow}(\tau) = \sum_{k=1}^{\infty}\left[\int_{-1}^0 I_k^{\downarrow}(\tau,\mu)\mu d\mu\right] + F_0 e^{-\tfrac{\tau}{\mu_0}}
$$  

---

## Physical formulas for SOS method with specular surface

## 1st order radiance field  

### Downward field  

**Upper atmospheric layer** ($z_{TOA} \geq z \geq z_{up}$):  

$$
I_1^{\downarrow}(\tau,\mu \le 0) = \frac{\mu_0}{\mu_0+\mu}\omega_{\text{atm}}P_{\text{atm}}(\mu,\mu_0)\frac{F_0}{4\pi}\left(e^{-\tfrac{\tau}{\mu_0}}-e^{\tfrac{\tau}{\mu}}\right) + \frac{\mu_0}{\mu_0-\mu}\omega_{\text{atm}}P_{\text{atm}}(\mu,-\mu_0)\frac{F_0 \rho_{\text{grd}} e^{-\tfrac{\tau_{\text{atm}}^{\ast}+\tau_{\text{aer}}^{\ast}}{\mu_0}}}{4\pi}\left(e^{-\tfrac{\tau_{\text{atm}}^{\ast}+\tau_{\text{aer}}^{\ast}-\tau}{\mu_0}}-e^{-\tfrac{\tau_{\text{atm}}^{\ast}+\tau_{\text{aer}}^{\ast}}{\mu_0}}e^{\tfrac{\tau}{\mu}}\right)
$$  

**Aerosol layer** ($z_{up} \geq z \geq z_{down}$):  

$$
I_1^{\downarrow}(\tau,\mu \le 0) = I_1^{\downarrow}(\tau_{up},\mu)e^{\tfrac{\tau-\tau_{up}}{\mu}} + \frac{\mu_0}{\mu_0+\mu}\left[\frac{d\tau_{\text{atm}}}{d\tau_{\text{atm}}+d\tau_{\text{aer}}}\omega_{\text{atm}}P_{\text{atm}}(\mu,\mu_0)+\frac{d\tau_{\text{aer}}}{d\tau_{\text{atm}}+d\tau_{\text{aer}}}\omega_{\text{aer}}P_{\text{aer}}(\mu,\mu_0)\right]\frac{F_0}{4\pi}\left(e^{-\tfrac{\tau}{\mu_0}}-e^{-\tfrac{\tau_{up}}{\mu_0}}e^{\tfrac{\tau-\tau_{up}}{\mu}}\right) +
$$

$$
\frac{\mu_0}{\mu_0-\mu}\left[\frac{d\tau_{\text{atm}}}{d\tau_{\text{atm}}+d\tau_{\text{aer}}}\omega_{\text{atm}}P_{\text{atm}}(\mu,-\mu_0)+\frac{d\tau_{\text{aer}}}{d\tau_{\text{atm}}+d\tau_{\text{aer}}}\omega_{\text{aer}}P_{\text{aer}}(\mu,-\mu_0)\right]\frac{F_0 \rho_{\text{grd}} e^{-\tfrac{\tau_{\text{atm}}^{\ast}+\tau_{\text{aer}}^{\ast}}{\mu_0}}}{4\pi}\left(e^{-\tfrac{\tau_{\text{atm}}^{\ast}+\tau_{\text{aer}}^{\ast}-\tau}{\mu_0}}-e^{-\tfrac{\tau_{\text{atm}}^{\ast}+\tau_{\text{aer}}^{\ast}-\tau_{up}}{\mu_0}}e^{\tfrac{\tau-\tau_{up}}{\mu}}\right)
$$  

**Bottom atmospheric layer** ($z_{down} \geq z \geq 0$):  

$$
I_1^{\downarrow}(\tau,\mu \le 0) = I_1^{\downarrow}(\tau_{down},\mu)e^{\tfrac{\tau-\tau_{down}}{\mu}} + \frac{\mu_0}{\mu_0+\mu}\omega_{\text{atm}}P_{\text{atm}}(\mu,\mu_0)\frac{F_0}{4\pi}\left(e^{-\tfrac{\tau}{\mu_0}}-e^{-\tfrac{\tau_{down}}{\mu_0}}e^{\tfrac{\tau-\tau_{down}}{\mu}}\right) + \frac{\mu_0}{\mu_0-\mu}\omega_{\text{atm}}P_{\text{atm}}(\mu,-\mu_0)\frac{F_0 \rho_{\text{grd}} e^{-\tfrac{\tau_{\text{atm}}^{\ast}+\tau_{\text{aer}}^{\ast}}{\mu_0}}}{4\pi}\left(e^{-\tfrac{\tau_{\text{atm}}^{\ast}+\tau_{\text{aer}}^{\ast}-\tau}{\mu_0}}-e^{-\tfrac{\tau_{\text{atm}}^{\ast}+\tau_{\text{aer}}^{\ast}-\tau_{down}}{\mu_0}}e^{\tfrac{\tau-\tau_{down}}{\mu}}\right)
$$  

### Upward field  

**Bottom atmospheric layer** ($z_{down} \geq z \geq 0$):  

$$
I_1^{\uparrow}(\tau,\mu \ge 0) = \rho_{\text{grd}}I_1^{\downarrow}(\tau_{\text{atm}}^{\ast}+\tau_{\text{aer}}^{\ast},-\mu)e^{-\tfrac{\tau_{\text{atm}}^{\ast}+\tau_{\text{aer}}^{\ast}-\tau}{\mu}} + \frac{\mu_0}{\mu_0+\mu}\omega_{\text{atm}}P_{\text{atm}}(\mu,\mu_0)\frac{F_0}{4\pi}\left(e^{-\tfrac{\tau}{\mu_0}}-e^{-\tfrac{\tau_{down}}{\mu_0}}e^{-\tfrac{\tau_{down}-\tau}{\mu}}\right) + \frac{\mu_0}{\mu_0-\mu}\omega_{\text{atm}}P_{\text{atm}}(\mu,-\mu_0)\frac{F_0 \rho_{\text{grd}} e^{-\tfrac{\tau_{\text{atm}}^{\ast}+\tau_{\text{aer}}^{\ast}}{\mu_0}}}{4\pi}\left(e^{-\tfrac{\tau_{\text{atm}}^{\ast}+\tau_{\text{aer}}^{\ast}-\tau}{\mu_0}}-e^{-\tfrac{\tau_{\text{atm}}^{\ast}+\tau_{\text{aer}}^{\ast}-\tau_{down}}{\mu_0}}e^{-\tfrac{\tau_{down}-\tau}{\mu}}\right)
$$  

**Aerosol layer** ($z_{up} \geq z \geq z_{down}$):  

$$
I_1^{\uparrow}(\tau,\mu \ge 0) = I_1^{\uparrow}(\tau_{down},\mu)e^{-\tfrac{\tau_{down}-\tau}{\mu}} + \frac{\mu_0}{\mu_0+\mu}\left[\frac{d\tau_{\text{atm}}}{d\tau_{\text{atm}}+d\tau_{\text{aer}}}\omega_{\text{atm}}P_{\text{atm}}(\mu,\mu_0)+\frac{d\tau_{\text{aer}}}{d\tau_{\text{atm}}+d\tau_{\text{aer}}}\omega_{\text{aer}}P_{\text{aer}}(\mu,\mu_0)\right]\frac{F_0}{4\pi}\left(e^{-\tfrac{\tau}{\mu_0}}-e^{-\tfrac{\tau_{down}}{\mu_0}}e^{-\tfrac{\tau_{down}-\tau}{\mu}}\right) +
$$

$$
\frac{\mu_0}{\mu_0-\mu}\left[\frac{d\tau_{\text{atm}}}{d\tau_{\text{atm}}+d\tau_{\text{aer}}}\omega_{\text{atm}}\,P_{\text{atm}}(\mu,-\mu_0)+\frac{d\tau_{\text{aer}}}{d\tau_{\text{atm}}+d\tau_{\text{aer}}}\omega_{\text{aer}}P_{\text{aer}}(\mu,-\mu_0)\right]\frac{F_0 \rho_{\text{grd}} e^{-\tfrac{\tau_{\text{atm}}^{\ast}+\tau_{\text{aer}}^{\ast}}{\mu_0}}}{4\pi}\left(e^{-\tfrac{\tau_{\text{atm}}^{\ast}+\tau_{\text{aer}}^{\ast}-\tau}{\mu_0}}-e^{-\tfrac{\tau_{\text{atm}}^{\ast}+\tau_{\text{aer}}^{\ast}-\tau_{down}}{\mu_0}}e^{-\tfrac{\tau_{down}-\tau}{\mu}}\right)
$$  

**Upper atmospheric layer** ($z_{TOA} \geq z \geq z_{up}$):  

$$
I_1^{\uparrow}(\tau,\mu \ge 0) = I_1^{\uparrow}(\tau_{up},\mu)e^{-\tfrac{\tau_{up}-\tau}{\mu}} + \frac{\mu_0}{\mu_0+\mu}\omega_{\text{atm}}P_{\text{atm}}(\mu,\mu_0)\frac{F_0}{4\pi}\left(e^{-\tfrac{\tau}{\mu_0}}-e^{-\tfrac{\tau_{up}}{\mu_0}}e^{-\tfrac{\tau_{up}-\tau}{\mu}}\right) + \frac{\mu_0}{\mu_0-\mu}\omega_{\text{atm}}P_{\text{atm}}(\mu,-\mu_0)\frac{F_0 \rho_{\text{grd}} e^{-\tfrac{\tau_{\text{atm}}^{\ast}+\tau_{\text{aer}}^{\ast}}{\mu_0}}}{4\pi}\left(e^{-\tfrac{\tau_{\text{atm}}^{\ast}+\tau_{\text{aer}}^{\ast}-\tau}{\mu_0}}-e^{-\tfrac{\tau_{\text{atm}}^{\ast}+\tau_{\text{aer}}^{\ast}-\tau_{up}}{\mu_0}}e^{-\tfrac{\tau_{up}-\tau}{\mu}}\right)
$$  

## $n \geq 2$ source function  

**Upper and bottom atmospheric layers** ($z_{TOA} \geq z \geq z_{up}$ and $z_{down} \geq z \geq 0$):  

$$
J_n(\tau,\mu) = \frac{\omega_{\text{atm}}}{4}\int_{-1}^1 P_{\text{atm}}(\mu,\mu')I_{n-1}^{(\uparrow\downarrow)}(\tau,\mu')d\mu'
$$  

**Aerosol layer** ($z_{up} \geq z \geq z_{down}$):  

$$
J_n(\tau,\mu) = \frac{d\tau_{\text{atm}}}{d\tau_{\text{atm}}+d\tau_{\text{aer}}}\frac{\omega_{\text{atm}}}{4}\int_{-1}^1 P_{\text{atm}}(\mu,\mu')I_{n-1}^{(\uparrow\downarrow)}(\tau,\mu')d\mu' + \frac{d\tau_{\text{aer}}}{d\tau_{\text{atm}}+d\tau_{\text{aer}}}\frac{\omega_{\text{aer}}}{4}\int_{-1}^1 P_{\text{aer}}(\mu,\mu')I_{n-1}^{(\uparrow\downarrow)}(\tau,\mu')d\mu'
$$  

---

## $n \geq 2$ radiance field  

### Downward field  

**Upper atmospheric layer** ($z_{TOA} \geq z \geq z_{up}$):  

$$
I_n^{\downarrow}(\tau, \mu \le 0) = -\int_0^{\tau} J_n(t,\mu)e^{\tfrac{\tau-t}{\mu}}\frac{dt}{\mu}
$$  

**Aerosol layer** ($z_{up} \geq z \geq z_{down}$):  

$$
I_n^{\downarrow}(\tau, \mu \le 0) = I_n^{\downarrow}(\tau_{up},\mu)e^{\tfrac{\tau-\tau_{up}}{\mu}} - \int_{\tau_{up}}^{\tau} J_n(t,\mu)e^{\tfrac{\tau-t}{\mu}}\frac{dt}{\mu}
$$  

**Bottom atmospheric layer** ($z_{down} \geq z \geq 0$):  

$$
I_n^{\downarrow}(\tau, \mu \le 0) = I_n^{\downarrow}(\tau_{down},\mu)e^{\tfrac{\tau-\tau_{down}}{\mu}} - \int_{\tau_{down}}^{\tau} J_n(t,\mu)e^{\tfrac{\tau-t}{\mu}}\frac{dt}{\mu}
$$  

### Upward field  

**Bottom atmospheric layer** ($z_{down} \geq z \geq 0$):  

$$
I_n^{\uparrow}(\tau, \mu \ge 0) = \rho_{\text{grd}}I_n^{\downarrow}(\tau_{\text{atm}}^{\ast}+\tau_{\text{aer}}^{\ast},-\,\mu)e^{-\tfrac{\tau_{\text{atm}}^{\ast}+\tau_{\text{aer}}^{\ast}-\tau}{\mu}} + \int_{\tau}^{\tau_{\text{atm}}^{\ast}+\tau_{\text{aer}}^{\ast}} J_n(t,\mu)e^{-\tfrac{t-\tau}{\mu}}\frac{dt}{\mu}
$$  

**Aerosol layer** ($z_{up} \geq z \geq z_{down}$):  

$$
I_n^{\uparrow}(\tau, \mu \ge 0) = I_n^{\uparrow}(\tau_{down},\mu)e^{-\tfrac{\tau_{down}-\tau}{\mu}} + \int_{\tau}^{\tau_{down}} J_n(t,\mu)e^{-\tfrac{t-\tau}{\mu}}\frac{dt}{\mu}
$$  

**Upper atmospheric layer** ($z_{TOA} \geq z \geq z_{up}$):  

$$
I_n^{\uparrow}(\tau, \mu \ge 0) = I_n^{\uparrow}(\tau_{up},\mu)e^{-\tfrac{\tau_{up}-\tau}{\mu}} + \int_{\tau}^{\tau_{up}} J_n(t,\mu)e^{-\tfrac{t-\tau}{\mu}}\frac{dt}{\mu}
$$  

---

## Total flux  

$$
F^{\uparrow}(\tau) = \sum_{k=1}^{\infty}\left[\int_0^1 I_k^{\uparrow}(\tau,\mu)\mu d\mu\right] + F_0 \rho_{\text{grd}} e^{-\tfrac{2(\tau_{\text{atm}}^{\ast}+\tau_{\text{aer}}^{\ast})-\tau}{\mu_0}}
$$  

$$
F^{\downarrow}(\tau) = \sum_{k=1}^{\infty}\left[\int_{-1}^0 I_k^{\downarrow}(\tau,\mu)\mu d\mu\right] + F_0 e^{-\tfrac{\tau}{\mu_0}}
$$  

