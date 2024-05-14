"""
The Energy balance models response to a periodic forcing Chapter 5.2 Figure 5.8
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)
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

driven_frequency = 0.01#0.01#0.03
years = 2000
skip_driven_years = 0

lats = np.flip(np.arange(0, 180, 2.5) - 88.75)
T0 = np.flip(np.load('./Input_data/Satellite_T_LAT.npy'))
S = np.flip(np.load('./Input_data/S_0BP.npy'))
normalized_s = S / np.sum(S)

def bring_to_equilibrium(T0):
    temp = np.zeros((50, 72))
    temp[0,:] = np.copy(T0)
    b = 0
    for i in range(50):
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

def driven_term(temp_array, year):
    # if year < skip_driven_years:
    #     return np.zeros(len(temp_array))
    #
    # year = year - skip_driven_years
    # baseline = np.ones(len(temp_array))
    baseline = normalized_s
    return E * np.sin(2 * np.pi * driven_frequency * year) * Area_Calc(72)# * baseline

def calculate_dT_dt(last_step, year):
    dT_dt_unscaled = (
        absorptive_term(last_step)
        - emissive_term(last_step)
        + convection_term(last_step)
        + driven_term(last_step, year)
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
    #plt.figure()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10,7.5),gridspec_kw={'height_ratios': [3,1]})
    fig.subplots_adjust(right = 0.88, top=0.98, left=0.135 ,hspace=0)
    trip_flag = False

    T_eq = bring_to_equilibrium(T0)
    T = np.copy(T_eq)
    T_list = [np.zeros(len(T))]
    E_list = [np.zeros(len(T))]
    driven_term_list = []
    north_ice_line_list = []
    south_ice_line_list = []

    n, s = calculate_ice_lims(T)
    north_ice_line_list.append(n)
    south_ice_line_list.append(s)

    for year in range(years):
        dT_dt = calculate_dT_dt(T, year)
        T = T + dT_dt
        # print(T.size, T0.size)
        T_list.append(area_aware_mean(T) - area_aware_mean(T_eq))
        E_list.append(area_aware_mean(driven_term(T, year)))

        area_averaged_driven_term = area_aware_mean(driven_term(T, year))

        n, s = calculate_ice_lims(T)
        n = min(n, 0.5)
        s = max(s, 0.5)
        if n == s and not trip_flag:
            print("Breakdown (W/m2)", area_averaged_driven_term)
            trip_flag = True

        north_ice_line_list.append(n)
        south_ice_line_list.append(s)

        driven_term_list.append(area_averaged_driven_term)
        if year > 0 and year % 100 == 0:
            E += 50

        # plt.plot(lats, T, label="Year {}".format(years))

    # plt.legend()
    ax1.text(45, 14.5, r"(a)")
    ax1.scatter(np.arange(years), driven_term_list, c='k', s=2)
    ax1.set_xlim((0, years))
    ax1.set_ylim(-17.5, 17.5)
    ax1.set_yticks(np.linspace(-15, 15, num=7, endpoint=True))
    ax1.yaxis.set_minor_locator(MultipleLocator(5))
    ax1.set_ylabel(r'Energy (W\,$\cdot$\,m$^{-2}$)')
    ax1.grid(True, which='major', linestyle='--', alpha=0.5)
    ax1.xaxis.set_ticklabels([])
    ax1.xaxis.set_ticks_position('none')
    ax1.title.set_visible(False)

    ax2.text(45,0.75, r"(b)")
    ax2.scatter(np.arange(years + 1), north_ice_line_list, c='b', s=2)
    ax2.scatter(np.arange(years + 1), south_ice_line_list, c='r', s=2)
    #ax2.yaxis.set_label_position("right")
    #ax2.yaxis.tick_right()
    ax2.set_yticks(np.linspace(0,1,num=5, endpoint=True))
    ax2.set_yticklabels(np.linspace(-90, 90, num=5, endpoint=True).astype(int))
    #ax2.yaxis.set_minor_locator(MultipleLocator(0.25))
    ax2.tick_params(axis='y', which='major', pad=15)
    ax2.set_ylabel(r'Ice Line Latitude'"\n"r'[degrees north]')
    ax2.set_ylim(0, 1)
    ax2.set_xlim((0, years))
    ax2.grid(True, which='major', linestyle='--', alpha=0.5)
    ax2.set_xlim((0, years))
    ax2.set_xticks(np.linspace(0, years, num=9, endpoint=True))
    ax2.xaxis.set_minor_locator(MultipleLocator(50))
    ax2.set_xlabel(r'Time (years)')
    ax2.title.set_visible(False)

    #ax3.text(45, -50, r"(c)")
    twin1 = ax1.twinx()
    twin1.scatter(np.arange(years), T_list[1:], c='r', s=2)
    twin1.set_yticks(np.linspace(-60, 60, num=5, endpoint=True))
    twin1.set_yticklabels(np.linspace(-60, 60, num=5, endpoint=True).astype(int))
    twin1.set_ylabel(r'Mean Temperature ($^\circ$C)')
    twin1.set_ylim(-70, 70)
    twin1.yaxis.set_label_position("right")
    twin1.yaxis.label.set_color("r")
    twin1.yaxis.tick_right()
    twin1.tick_params(axis='y', which='both', colors='r')
    twin1.yaxis.set_minor_locator(MultipleLocator(5))

    plt.savefig('./Forcing_Model_result v3.pdf', format='pdf', dpi=300)