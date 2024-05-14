"""
Chapter 5.2 Figure 5.6 Amplitude vs Frequency plot
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)
from matplotlib.ticker import FuncFormatter
from mpl_toolkits.axes_grid1.parasite_axes import SubplotHost
import matplotlib as mpl
mpl.rcParams['text.usetex'] = True
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.size'] = 22

A = 202.8
B = 2.17
C = 6.5
D = 3.81
E = 10
driven_frequency = 0.01

lats = np.flip(np.arange(0, 180, 2.5) - 88.75)
T0 = np.flip(np.load('./Input_data/Satellite_T_LAT.npy'))
S = np.flip(np.load('./Input_data/S_0BP.npy'))
S = S
def bring_to_equilibrium(T0):
    temp = np.zeros((100, 72))
    temp[0,:] = np.copy(T0)
    b = 0
    for i in range(100):
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

def absorptive_term(temp_array):
    return S * (1 - Alpha(temp_array))

def emissive_term(temp_array):
    return A + B * temp_array

def convection_term(temp_array):
    return D * (area_aware_mean(temp_array) * np.ones(len(temp_array)) - temp_array)

def driven_term(year, freq=driven_frequency, energy=E):
    #freq = 1/freq
    return energy * np.cos(2 * np.pi * freq * year) * Area_Calc(72)

def calculate_dT_dt(last_step, year, freq, energy):
    dT_dt_unscaled = (
        absorptive_term(last_step)
        - emissive_term(last_step)
        + convection_term(last_step)
        + driven_term(year, freq, energy)
    )
    return dT_dt_unscaled / C

if __name__ == "__main__":
    E_range = np.arange(100, 6000, 100)
    F_range = np.arange(0.0005, 0.1, 0.0005)
    E_list = np.zeros(len(E_range))
    mean_temp = np.zeros((len(E_range), len(F_range)))
    T_eq = bring_to_equilibrium(T0)
    for i, energy in enumerate(E_range):
        E_list[i] = area_aware_mean(driven_term(0, 1, energy))
        for j, freq in enumerate(F_range):
            T = np.copy(T_eq)
            #years=100
            #if freq > 0.001:
            years = int(2 * (1 / freq))
            T_list = np.zeros(years)
            for year in range(years):
                dT_dt = calculate_dT_dt(T, year, freq, energy)
                T = T + dT_dt
                T_list[year] = area_aware_mean(T)
            mean_temp[i, j] = float(np.mean(T_list) - area_aware_mean(T_eq))
        print(i)

    plt.close('all')
    fig = plt.figure(figsize=(10, 7.5))
    ax = SubplotHost(fig, 1,1,1)
    fig.add_subplot(ax)
    im = ax.imshow(
        mean_temp,
        cmap="Blues_r",
        vmax=1,
        vmin=-50,
        extent=[min(F_range), max(F_range), max(E_list), min(E_list)],
        aspect=max(F_range) / max(E_list),
    )
    cbar = fig.colorbar(im, ax=ax, ticks = np.linspace(0, -50, num=6, endpoint=True), label=r'Mean Temperature Anomaly ($^\circ$C)')
    cbar.ax.set_yticklabels(np.linspace(0, -50, num=6, endpoint=True).astype(int))

    ax.set_yticks(np.linspace(0, 100, num=11, endpoint=True))
    ax.set_ylabel(r"Amplitude (W\,$\cdot$\,m$^{-2}$)")
    ax.set_ylim(1.75, 101)
    ax.yaxis.set_minor_locator(MultipleLocator(2))
    ax.set_xticks(np.linspace(0.02, 0.1, num=5, endpoint=True))
    ax.set_xlabel("Frequency (years$^{-1}$)")
    ax.xaxis.set_minor_locator(MultipleLocator(0.01))
    twin1 = ax.twiny()
    twin1.set_xticks([0.005, 0.2, 0.5, 1])
    twin1.set_xticklabels([2000, 50, 20, 10])
    twin1.set_xlabel(r'Period (years)')
    twin1.xaxis.set_label_position("top")
    twin1.xaxis.tick_top()
    twin1.xaxis.set_minor_locator(MultipleLocator(5))

    plt.savefig('./Forcing_Model_frequency_result.pdf', format='pdf', dpi=300)


