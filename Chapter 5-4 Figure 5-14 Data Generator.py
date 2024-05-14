"""
The Energy balance model variance for each of the three cases T_Eq, T_D and T_s as presented in Chapter 5.4 Figure 5.14
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)
import math
import matplotlib as mpl
#mpl.rcParams['text.usetex'] = True
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
simulations = 10000

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
    Area = Area_Calc(72)
    #Subject each model to a random energy E
    print('Subjected to random energy every year at 2.5 Wm2')
    stddev = 150 #approx equal to 2.7 W/m2
    variance_T_Eq = np.zeros((simulations, 72))
    variance_T_S = np.zeros((simulations, 72))
    variance_T_D = np.zeros((simulations, 72))
    T_list_Eq = np.zeros((simulations, years, 72))
    T_list_S = np.zeros((simulations, years, 72))
    T_list_D = np.zeros((simulations, years, 72))
    sim = 0
    fsim = 0
    while (sim < simulations):
        T_Eq= np.copy(T_Eq_eq)
        T_S = np.copy(T_S_eq)
        T_D = np.copy(T_D_eq)
        E = np.random.default_rng().normal(0, stddev, size=years)
        for year in range(years):
            #Run model for T_Eq
            dT_dt = calculate_dT_dt(T_Eq, year, E[year], S=S, D=3.81)
            T_Eq = T_Eq + dT_dt
            T_list_Eq[sim, year, :] = T_Eq
            #Run model for T_S
            dT_dt = calculate_dT_dt(T_S, year, E[year], S=S*0.975, D=3.81)
            T_S = T_S + dT_dt
            T_list_S[sim, year, :] = T_S
            #Run model for T_D
            dT_dt = calculate_dT_dt(T_D, year, E[year], S=S, D=3.25)
            T_D = T_D + dT_dt
            T_list_D[sim, year, :] = T_D

        variance_T_Eq[sim, :] = np.var(T_list_Eq[sim, :, :], axis=0, ddof=1)
        variance_T_S[sim, :] = np.var(T_list_S[sim, :, :], axis=0, ddof=1)
        variance_T_D[sim, :] = np.var(T_list_D[sim, :, :], axis=0, ddof=1)
        sim += 1
    np.save('R_2_5_T_Eq.npy', variance_T_Eq)
    np.save('R_2_5_T_S.npy', variance_T_S)
    np.save('R_2_5_T_D.npy', variance_T_D)

    #Subject to random applitude periodic forcing
    print('Subject to random amplitude perdiodic forcing at 2.5 Wm2')
    stddev = 200  # approx equal to 2.7 W/m2
    variance_T_Eq = np.zeros((simulations, 72))
    variance_T_S = np.zeros((simulations, 72))
    variance_T_D = np.zeros((simulations, 72))
    T_list_Eq = np.zeros((simulations, years, 72))
    T_list_S = np.zeros((simulations, years, 72))
    T_list_D = np.zeros((simulations, years, 72))
    sim = 0
    fsim = 0
    while (sim < simulations):
        T_Eq = np.copy(T_Eq_eq)
        T_S = np.copy(T_S_eq)
        T_D = np.copy(T_D_eq)
        E = np.random.default_rng().normal(0, stddev, size=int(years/period + 1))
        decade = 0
        for year in range(years):
            if year%period == 0:
                decade += 1
            # Run model for T_Eq
            dT_dt = calculate_dT_dt(T_Eq, year, E[decade], S=S, D=3.81, periodic=2)
            T_Eq = T_Eq + dT_dt
            T_list_Eq[sim, year, :] = T_Eq
            # Run model for T_S
            dT_dt = calculate_dT_dt(T_S, year, E[decade], S=S * 0.975, D=3.81, periodic=2)
            T_S = T_S + dT_dt
            T_list_S[sim, year, :] = T_S
            # Run model for T_D
            dT_dt = calculate_dT_dt(T_D, year, E[decade], S=S, D=3.25, periodic=2)
            T_D = T_D + dT_dt
            T_list_D[sim, year, :] = T_D

        variance_T_Eq[sim, :] = np.var(T_list_Eq[sim, :, :], axis=0, ddof=1)
        variance_T_S[sim, :] = np.var(T_list_S[sim, :, :], axis=0, ddof=1)
        variance_T_D[sim, :] = np.var(T_list_D[sim, :, :], axis=0, ddof=1)
        sim += 1

    np.save('P_2_5_T_Eq.npy', variance_T_Eq)
    np.save('P_2_5_T_S.npy', variance_T_S)
    np.save('P_2_5_T_D.npy', variance_T_D)

    print('Subjected to random energy every year at 5 Wm2')
    stddev = 300 #approx equal to 2.7 W/m2
    variance_T_Eq = np.zeros((simulations, 72))
    variance_T_S = np.zeros((simulations, 72))
    variance_T_D = np.zeros((simulations, 72))
    T_list_Eq = np.zeros((simulations, years, 72))
    T_list_S = np.zeros((simulations, years, 72))
    T_list_D = np.zeros((simulations, years, 72))
    sim = 0
    fsim = 0
    while (sim < simulations):
        T_Eq= np.copy(T_Eq_eq)
        T_S = np.copy(T_S_eq)
        T_D = np.copy(T_D_eq)
        E = np.random.default_rng().normal(0, stddev, size=years)
        for year in range(years):
            #Run model for T_Eq
            dT_dt = calculate_dT_dt(T_Eq, year, E[year], S=S, D=3.81)
            T_Eq = T_Eq + dT_dt
            T_list_Eq[sim, year, :] = T_Eq
            #Run model for T_S
            dT_dt = calculate_dT_dt(T_S, year, E[year], S=S*0.975, D=3.81)
            T_S = T_S + dT_dt
            T_list_S[sim, year, :] = T_S
            #Run model for T_D
            dT_dt = calculate_dT_dt(T_D, year, E[year], S=S, D=3.25)
            T_D = T_D + dT_dt
            T_list_D[sim, year, :] = T_D

        variance_T_Eq[sim, :] = np.var(T_list_Eq[sim, :, :], axis=0, ddof=1, ddof=1)
        variance_T_S[sim, :] = np.var(T_list_S[sim, :, :], axis=0, ddof=1)
        variance_T_D[sim, :] = np.var(T_list_D[sim, :, :], axis=0, ddof=1)
        sim += 1

    np.save('R_5_T_Eq.npy', variance_T_Eq)
    np.save('R_5_T_S.npy', variance_T_S)
    np.save('R_5_T_D.npy', variance_T_D)

    #Subject to random applitude periodic forcing
    print('Subject to random amplitude perdiodic forcing at 5 Wm2')
    stddev = 400  # approx equal to 2.7 W/m2
    variance_T_Eq = np.zeros((simulations, 72))
    variance_T_S = np.zeros((simulations, 72))
    variance_T_D = np.zeros((simulations, 72))
    T_list_Eq = np.zeros((simulations, years, 72))
    T_list_S = np.zeros((simulations, years, 72))
    T_list_D = np.zeros((simulations, years, 72))
    sim = 0
    fsim = 0
    fsim_S = 0
    fsim_D = 0
    fsim_Eq = 0
    while (sim < simulations):
        T_Eq = np.copy(T_Eq_eq)
        T_S = np.copy(T_S_eq)
        T_D = np.copy(T_D_eq)
        E = np.random.default_rng().normal(0, stddev, size=int(years/period + 1))
        decade = 0
        for year in range(years):
            if year%period == 0:
                decade += 1
            # Run model for T_Eq
            dT_dt = calculate_dT_dt(T_Eq, year, E[decade], S=S, D=3.81, periodic=2)
            T_Eq = T_Eq + dT_dt
            T_list_Eq[sim, year, :] = T_Eq
            # Run model for T_S
            dT_dt = calculate_dT_dt(T_S, year, E[decade], S=S * 0.975, D=3.81, periodic=2)
            T_S = T_S + dT_dt
            T_list_S[sim, year, :] = T_S
            # Run model for T_D
            dT_dt = calculate_dT_dt(T_D, year, E[decade], S=S, D=3.25, periodic=2)
            T_D = T_D + dT_dt
            T_list_D[sim, year, :] = T_D

        variance_T_Eq[sim, :] = np.var(T_list_Eq[sim, :, :], axis=0, ddof=1)
        variance_T_S[sim, :] = np.var(T_list_S[sim, :, :], axis=0, ddof=1)
        variance_T_D[sim, :] = np.var(T_list_D[sim, :, :], axis=0, ddof=1)
        sim += 1

    np.save('P_5_T_Eq.npy', variance_T_Eq)
    np.save('P_5_T_S.npy', variance_T_S)
    np.save('P_5_T_D.npy', variance_T_D)

"""
Ice free solution code
"""
    print('Subjected to random energy every year at 2.5 Wm2, ice free')
    stddev = 150  # approx equal to 2.7 W/m2
    variance_T_Eq = np.zeros((simulations, 72))
    variance_T_S = np.zeros((simulations, 72))
    variance_T_D = np.zeros((simulations, 72))
    T_list_Eq = np.zeros((simulations, years, 72))
    T_list_S = np.zeros((simulations, years, 72))
    T_list_D = np.zeros((simulations, years, 72))
    sim = 0
    fsim = 0
    fsim_S = 0
    fsim_D = 0
    fsim_Eq = 0
    while (sim < simulations):
        T_Eq = np.copy(T_Eq_eq)
        T_S = np.copy(T_S_eq)
        T_D = np.copy(T_D_eq)
        E = np.random.default_rng().normal(0, stddev, size=years)
        for year in range(years):
            # Run model for T_Eq
            dT_dt = calculate_dT_dt(T_Eq, year, E[year], S=S, D=3.81)
            T_Eq = T_Eq + dT_dt
            T_list_Eq[sim, year, :] = T_Eq
            # Run model for T_S
            dT_dt = calculate_dT_dt(T_S, year, E[year], S=S * 0.975, D=3.81)
            T_S = T_S + dT_dt
            T_list_S[sim, year, :] = T_S
            # Run model for T_D
            dT_dt = calculate_dT_dt(T_D, year, E[year], S=S, D=3.25)
            T_D = T_D + dT_dt
            T_list_D[sim, year, :] = T_D
        check_Eq = np.average(T_list_Eq[sim, :, :], axis=1, weights=Area_Calc(72))
        check_S = np.average(T_list_Eq[sim, :, :], axis=1, weights=Area_Calc(72))
        check_D = np.average(T_list_Eq[sim, :, :], axis=1, weights=Area_Calc(72))
        if np.any(check_Eq < -9) or np.any(check_S < -9) or np.any(check_D < -9):
            fsim += 1
            if np.any(check_Eq < -9):
                fsim_Eq += 1
            if np.any(check_S < -9):
                fsim_S += 1
            if np.any(check_D < -9):
                fsim_D += 1
        else:  # Compute the variances
            variance_T_Eq[sim, :] = np.var(T_list_Eq[sim, :, :], axis=0, ddof=1)
            variance_T_S[sim, :] = np.var(T_list_S[sim, :, :], axis=0, ddof=1)
            variance_T_D[sim, :] = np.var(T_list_D[sim, :, :], axis=0, ddof=1)
            sim += 1

    print("Random 2.5W total fails = ", fsim)
    print("Random 2.5W Eq fails = ", fsim_Eq)
    print("Random 2.5W S fails = ", fsim_S)
    print("Random 2.5W D fails = ", fsim_D)
    np.save('R_2_5_T_Eq_Ice.npy', variance_T_Eq)
    np.save('R_2_5_T_S_Ice.npy', variance_T_S)
    np.save('R_2_5_T_D_Ice.npy', variance_T_D)

    # Subject to random applitude periodic forcing
    print('Subject to random amplitude perdiodic forcing at 2.5 Wm2 ice free')
    stddev = 200  # approx equal to 2.7 W/m2
    variance_T_Eq = np.zeros((simulations, 72))
    variance_T_S = np.zeros((simulations, 72))
    variance_T_D = np.zeros((simulations, 72))
    T_list_Eq = np.zeros((simulations, years, 72))
    T_list_S = np.zeros((simulations, years, 72))
    T_list_D = np.zeros((simulations, years, 72))
    sim = 0
    fsim = 0
    fsim_S = 0
    fsim_D = 0
    fsim_Eq = 0
    while (sim < simulations):
        T_Eq = np.copy(T_Eq_eq)
        T_S = np.copy(T_S_eq)
        T_D = np.copy(T_D_eq)
        E = np.random.default_rng().normal(0, stddev, size=int(years / period + 1))
        decade = 0
        for year in range(years):
            if year % period == 0:
                decade += 1
            # Run model for T_Eq
            dT_dt = calculate_dT_dt(T_Eq, year, E[decade], S=S, D=3.81, periodic=2)
            T_Eq = T_Eq + dT_dt
            T_list_Eq[sim, year, :] = T_Eq
            # Run model for T_S
            dT_dt = calculate_dT_dt(T_S, year, E[decade], S=S * 0.975, D=3.81, periodic=2)
            T_S = T_S + dT_dt
            T_list_S[sim, year, :] = T_S
            # Run model for T_D
            dT_dt = calculate_dT_dt(T_D, year, E[decade], S=S, D=3.25, periodic=2)
            T_D = T_D + dT_dt
            T_list_D[sim, year, :] = T_D
        check_Eq = np.average(T_list_Eq[sim, :, :], axis=1, weights=Area_Calc(72))
        check_S = np.average(T_list_Eq[sim, :, :], axis=1, weights=Area_Calc(72))
        check_D = np.average(T_list_Eq[sim, :, :], axis=1, weights=Area_Calc(72))
        if np.any(check_Eq < -9) or np.any(check_S < -9) or np.any(check_D < -9):
            fsim += 1
            if np.any(check_Eq < -9):
                fsim_Eq += 1
            if np.any(check_S < -9):
                fsim_S += 1
            if np.any(check_D < -9):
                fsim_D += 1
        else:  # Compute the variances
            variance_T_Eq[sim, :] = np.var(T_list_Eq[sim, :, :], axis=0, ddof=1)
            variance_T_S[sim, :] = np.var(T_list_S[sim, :, :], axis=0, ddof=1)
            variance_T_D[sim, :] = np.var(T_list_D[sim, :, :], axis=0, ddof=1)
            sim += 1

    print("Periodic 2.5W total fails = ", fsim)
    print("Periodic 2.5W Eq fails = ", fsim_Eq)
    print("Periodic 2.5W S fails = ", fsim_S)
    print("Periodic 2.5W D fails = ", fsim_D)
    #
    np.save('P_2_5_T_Eq_Ice.npy', variance_T_Eq)
    np.save('P_2_5_T_S_Ice.npy', variance_T_S)
    np.save('P_2_5_T_D_Ice.npy', variance_T_D)

    print('Subjected to random energy every year at 5 Wm2 ice free')
    stddev = 300  # approx equal to 2.7 W/m2
    variance_T_Eq = np.zeros((simulations, 72))
    variance_T_S = np.zeros((simulations, 72))
    variance_T_D = np.zeros((simulations, 72))
    T_list_Eq = np.zeros((simulations, years, 72))
    T_list_S = np.zeros((simulations, years, 72))
    T_list_D = np.zeros((simulations, years, 72))
    sim = 0
    fsim = 0
    fsim_S = 0
    fsim_D = 0
    fsim_Eq = 0
    while (sim < simulations):
        T_Eq = np.copy(T_Eq_eq)
        T_S = np.copy(T_S_eq)
        T_D = np.copy(T_D_eq)
        E = np.random.default_rng().normal(0, stddev, size=years)
        for year in range(years):
            # Run model for T_Eq
            dT_dt = calculate_dT_dt(T_Eq, year, E[year], S=S, D=3.81)
            T_Eq = T_Eq + dT_dt
            T_list_Eq[sim, year, :] = T_Eq
            # Run model for T_S
            dT_dt = calculate_dT_dt(T_S, year, E[year], S=S * 0.975, D=3.81)
            T_S = T_S + dT_dt
            T_list_S[sim, year, :] = T_S
            # Run model for T_D
            dT_dt = calculate_dT_dt(T_D, year, E[year], S=S, D=3.25)
            T_D = T_D + dT_dt
            T_list_D[sim, year, :] = T_D
        check_Eq = np.average(T_list_Eq[sim, :, :], axis=1, weights=Area_Calc(72))
        check_S = np.average(T_list_Eq[sim, :, :], axis=1, weights=Area_Calc(72))
        check_D = np.average(T_list_Eq[sim, :, :], axis=1, weights=Area_Calc(72))
        if np.any(check_Eq < -9) or np.any(check_S < -9) or np.any(check_D < -9):
            fsim += 1
            if np.any(check_Eq < -9):
                fsim_Eq += 1
            if np.any(check_S < -9):
                fsim_S += 1
            if np.any(check_D < -9):
                fsim_D += 1
        else:  # Compute the variances
            variance_T_Eq[sim, :] = np.var(T_list_Eq[sim, :, :], axis=0, ddof=1)
            variance_T_S[sim, :] = np.var(T_list_S[sim, :, :], axis=0, ddof=1)
            variance_T_D[sim, :] = np.var(T_list_D[sim, :, :], axis=0, ddof=1)
            sim += 1

    print("Random 5W total fails = ", fsim)
    print("Random 5W Eq fails = ", fsim_Eq)
    print("Random 5W S fails = ", fsim_S)
    print("Random 5W D fails = ", fsim_D)
    np.save('R_5_T_Eq_Ice.npy', variance_T_Eq)
    np.save('R_5_T_S_Ice.npy', variance_T_S)
    np.save('R_5_T_D_Ice.npy', variance_T_D)

    # Subject to random applitude periodic forcing
    print('Subject to random amplitude perdiodic forcing at 5 Wm2 ice free')
    stddev = 400  # approx equal to 2.7 W/m2
    variance_T_Eq = np.zeros((simulations, 72))
    variance_T_S = np.zeros((simulations, 72))
    variance_T_D = np.zeros((simulations, 72))
    T_list_Eq = np.zeros((simulations, years, 72))
    T_list_S = np.zeros((simulations, years, 72))
    T_list_D = np.zeros((simulations, years, 72))
    sim = 0
    fsim = 0
    fsim_S = 0
    fsim_D = 0
    fsim_Eq = 0
    while (sim < simulations):
        T_Eq = np.copy(T_Eq_eq)
        T_S = np.copy(T_S_eq)
        T_D = np.copy(T_D_eq)
        E = np.random.default_rng().normal(0, stddev, size=int(years / period + 1))
        decade = 0
        for year in range(years):
            if year % period == 0:
                decade += 1
            # Run model for T_Eq
            dT_dt = calculate_dT_dt(T_Eq, year, E[decade], S=S, D=3.81, periodic=2)
            T_Eq = T_Eq + dT_dt
            T_list_Eq[sim, year, :] = T_Eq
            # Run model for T_S
            dT_dt = calculate_dT_dt(T_S, year, E[decade], S=S * 0.975, D=3.81, periodic=2)
            T_S = T_S + dT_dt
            T_list_S[sim, year, :] = T_S
            # Run model for T_D
            dT_dt = calculate_dT_dt(T_D, year, E[decade], S=S, D=3.25, periodic=2)
            T_D = T_D + dT_dt
            T_list_D[sim, year, :] = T_D
        #Check the model run didn't go ice covered
        check_Eq = np.average(T_list_Eq[sim, :, :], axis=1, weights=Area_Calc(72))
        check_S = np.average(T_list_Eq[sim, :, :], axis=1, weights=Area_Calc(72))
        check_D = np.average(T_list_Eq[sim, :, :], axis=1, weights=Area_Calc(72))
        if np.any(check_Eq < -9) or np.any(check_S < -9) or np.any(check_D < -9):
            fsim += 1
            if np.any(check_Eq < -9):
                fsim_Eq += 1
            if np.any(check_S < -9):
                fsim_S += 1
            if np.any(check_D < -9):
                fsim_D += 1
        else:#Compute the variances
            variance_T_Eq[sim, :] = np.var(T_list_Eq[sim, :, :], axis=0, ddof=1)
            variance_T_S[sim, :] = np.var(T_list_S[sim, :, :], axis=0, ddof=1)
            variance_T_D[sim, :] = np.var(T_list_D[sim, :, :], axis=0, ddof=1)
            sim += 1

    print("Periodic 5W total fails = ", fsim)
    print("Periodic 5W Eq fails = ", fsim_Eq)
    print("Periodic 5W S fails = ", fsim_S)
    print("Periodic 5W D fails = ", fsim_D)
    np.save('P_5_T_Eq_Ice.npy', variance_T_Eq)
    np.save('P_5_T_S_Ice.npy', variance_T_S)
    np.save('P_5_T_D_Ice.npy', variance_T_D)
