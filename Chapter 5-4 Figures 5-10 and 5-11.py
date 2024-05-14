"""
The Energy balance model results for each of the test cases in Chapter 5.4 Figures 5.10 and 5.11
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)
import math
import matplotlib as mpl
mpl.rcParams['text.usetex'] = True
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.size'] = 22

plt.close('all')
A = 202.8
B = 2.17
C = 6.5
D = 3.81
E = 0#3000  # Watts per radian

driven_frequency = 0.05#0.01#0.03
period = int(1/driven_frequency)
years = 1000
skip_driven_years = 0
simulations = 100

lats = np.flip(np.arange(0, 180, 2.5) - 88.75)
T0 = np.flip(np.load('./Input_data/Satellite_T_LAT.npy'))
S = np.flip(np.load('./Input_data/S_0BP.npy'))

def def_fn(X):
    N = int((X - 1) / 2)
    def fn(n):
        return math.factorial(2 * N) / (math.factorial(N + n) * math.factorial(N - n))
    n_list = np.arange(-N, N, 1)
    f_n = np.zeros(len(n_list))
    for i, n in enumerate(n_list):
        f_n[i] = fn(n)
    return f_n

def binomial_running_mean(temperature, f_n, X=21):
    N = int((X-1)/2)
    runningmean = np.zeros(len(temperature))
    n_list = np.arange(-N, N, 1)
    for k in range(N, int(len(temperature)-N)):
        numerator = np.zeros((int(2*N+1)))
        denominator = np.zeros((int(2*N+1)))
        for i, n in enumerate(n_list):
            numerator[i] = f_n[n] * temperature[int(k + n)]
            denominator[i] = f_n[n]
        runningmean[k] = np.sum(numerator) / np.sum(denominator)

    difference = temperature[N:int(len(temperature)-N)] - runningmean[N:int(len(temperature)-N)]
    difference = difference ** 2
    X = np.arange(N, int(len(temperature)-N))
    return X, runningmean[N:int(len(temperature)-N)], difference

def bring_to_equilibrium(T0, S, D=3.81):
    temp = np.zeros((99, 72))
    temp[0,:] = np.copy(T0)
    b = 0
    for i in range(99):
        X = np.average(temp[i, :], weights=Area_Calc(72)) - temp[i, :]
        dT = (S*1)*(1-Alpha(temp[i,:])) - (A + B*temp[i,:]) + D*X
        temp[i+1,:] = temp[i,:] + dT/C
        if abs(np.average(temp[i,:]-temp[i-1,:], weights=Area_Calc(72))) < 0.005:
            b += 1
            if b == 3:
                print(i)
                break
        else:
            b=0
    print(np.average(temp[i,:], weights=Area_Calc(72)))
    return temp[i,:]

def Alpha(T):
    return 0.5 - 0.2 * np.tanh((T + 8.8) / 10)

def Area_Calc(Number_Lats, Area_or_Weights="Weights"):
    if type(Number_Lats) == np.ndarray:
        if Number_Lats.ndim > 1:
            print("Latitude provided as 2d array")
            print("Attempting to flatten")
            Number_Lats = Number_Lats.flatten()
        Number_Lats = int(len(Number_Lats))

    if isinstance(Number_Lats, (int, float)) == True:
        if Number_Lats % 2 == 0:
            Lats = np.linspace(0, 90, num=int(Number_Lats / 2) + 1, endpoint=True) * np.pi / 180
            A = np.abs(2 * np.pi * 6.371e6**2 * np.diff(np.sin(Lats)))
            A = np.concatenate([A[::-1], A])
        else:
            Lats = np.linspace(90 / Number_Lats, 90, num=int((Number_Lats + 1) / 2), endpoint=True) * np.pi / 180
            A = np.abs(2 * np.pi * 6.371e6**2 * np.diff(np.sin(Lats)))
            A = np.concatenate([A[::-1], A])
            A = np.insert(A, int(len(A) / 2), 2 * np.abs(2 * np.pi * 6.371e6**2 * (np.sin(Lats[0]))))
    else:
        raise TypeError("Number_Lats must be a vector or an integer")

    if Area_or_Weights == "Area":
        return A
    elif Area_or_Weights == "Weights":
        return A / np.sum(A)
    else:
        raise ValueError('"Area" or "Weights" string specified incorrectly')

def area_aware_mean(temp_array):
    return np.average(temp_array, weights=Area_Calc(lats))

def absorptive_term(temp_array, S):
    return S * (1 - Alpha(temp_array))

def emissive_term(temp_array):
    return A + B * temp_array

def convection_term(temp_array, D):
    return D * (area_aware_mean(temp_array) * np.ones(len(temp_array)) - temp_array)

def driven_term(temp_array, year, E, periodic=0):
    if periodic == 0:
        Z = E * Area_Calc(72)
    else:
        Z = E * np.sin(2 * np.pi * driven_frequency * year) * Area_Calc(72)
    return Z

def calculate_dT_dt(last_step, year, E, S, D=3.81, periodic=0):
    dT_dt_unscaled = (
        absorptive_term(last_step, S)
        - emissive_term(last_step)
        + convection_term(last_step, D)
        + driven_term(last_step, year, E, periodic)
    )
    return dT_dt_unscaled / C

def calculate_ice_lims(temp_array):
    # The temp array is the temperatures at evenly spaced angles
    # Whenever the temperature is below -10 it is under ice

    north_hemisphere_ice_line_index = 0
    southern_hemisphere_ice_line_index = 0
    for i, t in enumerate(temp_array):
        if t < -10:
            north_hemisphere_ice_line_index = i
        else:
            break

    for i, t in enumerate(temp_array[::-1]):
        if t < -10:
            southern_hemisphere_ice_line_index = i
        else:
            break

    return [north_hemisphere_ice_line_index / len(temp_array), 1 - southern_hemisphere_ice_line_index / len(temp_array)]

if __name__ == "__main__":
    #Bring our 3 models to equilibrium
    T_Eq_eq = bring_to_equilibrium(T0, S)
    T_S_eq = bring_to_equilibrium(T0, S*0.975)
    T_D_eq = bring_to_equilibrium(T0, S, D=3.25)

"""
Figure 5.10 The equilibrium results of the EBM for each of the 3 test cases
"""

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10,7.5))
    fig.subplots_adjust(right=0.855, top=0.98, left=0.15, hspace=0)
    ax1.scatter(lats, Alpha(T_Eq_eq), c='k', s=2)
    ax1.scatter(lats, Alpha(T_D_eq), c='r', s=2)
    ax1.scatter(lats, Alpha(T_S_eq), c='b', s=2)
    ax1.set_xticks(np.flip(np.linspace(90, -90, num=5, endpoint=True)))
    ax1.xaxis.set_ticklabels([])
    ax1.set_xlim((90, -90))
    ax1.xaxis.set_ticks_position('none')
    ax1.title.set_visible(False)
    ax1.grid(True, which='major', linestyle='--', alpha=0.5)
    ax1.text(87, 0.7, r'(a)', color='k')
    ax1.set_ylim(0.2, 0.8)
    ax1.set_yticks(np.linspace(0.2,0.8, num=4, endpoint=True))
    ax1.yaxis.set_minor_locator(MultipleLocator(0.1))
    ax1.set_ylabel(r'Albedo')

    ax2.scatter(lats, emissive_term(T_Eq_eq), c='k', s=2, label='T$_{Eq}$')
    ax2.scatter(lats, emissive_term(T_S_eq), c='b', s=2, label='T$_{S}$')
    ax2.scatter(lats, emissive_term(T_D_eq), c='r', s=2, label='T$_{D}$')
    ax2.set_xticks(np.flip(np.linspace(90, -90, num=5, endpoint=True)))
    ax2.xaxis.set_ticklabels([])
    ax2.set_xlim((90, -90))
    ax2.xaxis.set_ticks_position('none')
    ax2.title.set_visible(False)
    ax2.grid(True, which='major', linestyle='--', alpha=0.5)
    ax2.text(87, 240, r'(b)', color='k')
    ax2.set_ylim(150, 260)
    ax2.set_yticks(np.linspace(150, 250, num=3, endpoint=True))
    ax2.yaxis.set_label_position("right")
    ax2.yaxis.tick_right()
    ax2.yaxis.set_minor_locator(MultipleLocator(10))
    ax2.legend(loc='center', markerscale=2., fancybox=False, edgecolor='k', prop={'size': 16})
    ax2.set_ylabel(r'OLR'"\n"r'(W\,$\cdot$\,m$^{-2}$)')

    ax3.scatter(lats, -convection_term(T_Eq_eq, D=3.81), c='k', s=2)
    ax3.scatter(lats, -convection_term(T_D_eq, D=3.25), c='r', s=2)
    ax3.scatter(lats, -convection_term(T_S_eq, D=3.81), c='b', s=2)
    ax3.set_xlim((90, -90))
    ax3.title.set_visible(False)
    ax3.grid(True, which='major', linestyle='--', alpha=0.5)
    ax3.text(87, 25, r'(c)', color='k')
    ax3.set_ylim(-110, 50)
    ax3.set_yticks(np.linspace(-100, 50, num=4, endpoint=True))
    ax3.yaxis.set_minor_locator(MultipleLocator(10))
    ax3.set_ylabel(r'Latitudinal Transport'"\n"r'(W\,$\cdot$\,m$^{-2}$)')
    ax3.set_xticks(np.flip(np.linspace(90, -90, num = 5, endpoint = True)))
    ax3.xaxis.set_minor_locator(MultipleLocator(5))
    ax3.set_xlabel('Latitude [degrees north]')

    plt.savefig('./Components_Eq_S_D.pdf', format='pdf', dpi=300)

"""
Figure 5.11 examples of each of the 4 types of forcing applied to the EBM
"""

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(10,7.5))
    fig.subplots_adjust(right=0.865, top=0.98, left=0.135, hspace=0)
    E_300 = np.random.default_rng().normal(0, 300, size=years)
    E_150 = np.random.default_rng().normal(0, 150, size=years)
    E_global_300 = np.zeros((years))
    E_global_150 = np.zeros((years))
    for i in range(years):
        E_global_300[i] = np.average(E_300[i] * Area_Calc(72), weights=Area_Calc(72))
        E_global_150[i] = np.average(E_150[i] * Area_Calc(72), weights=Area_Calc(72))
    ax1.scatter(np.arange(years), E_global_150, c='k', s=2)
    ax1.set_xlim(0, 1000)
    ax1.set_ylim(-10, 10)
    ax1.set_ylabel(r'Amplitude'"\n"r'(W\,$\cdot$\,m$^{-2}$)')
    ax1.set_yticks(np.linspace(-10, 10, num=3, endpoint=True))
    ax1.yaxis.set_minor_locator(MultipleLocator(2))
    ax1.xaxis.set_ticklabels([])
    ax1.xaxis.set_ticks_position('none')
    ax1.title.set_visible(False)
    ax1.text(10, 6, r'(a)')
    ax1.grid(True, which='major', linestyle='--', alpha=0.5)
    ax2.scatter(np.arange(years), E_global_300, c='k', s=2)
    ax2.set_xlim(0, 1000)
    ax2.set_ylim(-20, 20)
    ax2.set_yticks(np.linspace(-20, 20, num=3, endpoint=True))
    ax2.yaxis.set_label_position("right")
    ax2.set_ylabel(r'Amplitude'"\n"r'(W\,$\cdot$\,m$^{-2}$)')
    ax2.yaxis.tick_right()
    ax2.yaxis.set_minor_locator(MultipleLocator(5))
    ax2.xaxis.set_ticklabels([])
    ax2.xaxis.set_ticks_position('none')
    ax2.title.set_visible(False)
    ax2.text(10, 12, r'(b)')
    ax2.grid(True, which='major', linestyle='--', alpha=0.5)

    E_P_300 = np.random.default_rng().normal(0, 430, size=int(years / period + 1))
    E_P_150 = np.random.default_rng().normal(0, 200, size=int(years / period + 1))
    E_global_P_300 = np.zeros((years))
    E_global_P_150 = np.zeros((years))
    decade = 0
    for i in range(years):
        if i % period == 0:
            decade += 1
        E_global_P_300[i] = np.average(E_P_300[decade] * np.sin(2 * np.pi * driven_frequency * i) * Area_Calc(72), weights=Area_Calc(72))
        E_global_P_150[i] = np.average(E_P_150[decade] * np.sin(2 * np.pi * driven_frequency * i) * Area_Calc(72), weights=Area_Calc(72))
    ax3.plot(np.arange(years), E_global_P_150, c='k')
    ax3.set_xlim(0, 1000)
    ax3.set_ylim(-10, 10)
    ax3.set_ylabel(r'Amplitude'"\n"r'(W\,$\cdot$\,m$^{-2}$)')
    ax3.set_yticks(np.linspace(-10, 10, num=3, endpoint=True))
    ax3.yaxis.set_minor_locator(MultipleLocator(2))
    ax3.xaxis.set_ticklabels([])
    ax3.xaxis.set_ticks_position('none')
    ax3.title.set_visible(False)
    ax3.text(10, 6, r'(c)')
    ax3.grid(True, which='major', linestyle='--', alpha=0.5)
    ax4.plot(np.arange(years), E_global_P_300, c='k')
    ax4.set_xlim(0, 1000)
    ax4.set_ylim(-20, 20)
    ax4.set_yticks(np.linspace(-20, 20, num=3, endpoint=True))
    ax4.yaxis.set_label_position("right")
    ax4.set_ylabel(r'Amplitude'"\n"r'(W\,$\cdot$\,m$^{-2}$)')
    ax4.yaxis.tick_right()
    ax4.yaxis.set_minor_locator(MultipleLocator(5))
    ax4.xaxis.set_minor_locator(MultipleLocator(50))
    ax4.set_xlabel('Time (years)')
    ax4.text(10, 12, r'(d)')
    ax4.grid(True, which='major', linestyle='--', alpha=0.5)

    plt.savefig('./Random Forcing_Eq_S_D.pdf', format='pdf', dpi=300)

