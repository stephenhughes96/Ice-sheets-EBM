"""
This code is designed to solve the equilibrium equation model (change in temperature with time is 0)
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
def Area_Calc(Number_Lats, Area_or_Weights='Weights'):
    if type(Number_Lats) == np.ndarray:
        if Number_Lats.ndim > 1:
            print("Latitude provided as 2d array")
            print("Attempting to flatten")
            Number_Lats = Number_Lats.flatten()
        Number_Lats = int(len(Number_Lats))

    if isinstance(Number_Lats, (int, float)) == True:
        if Number_Lats % 2 == 0:
            Lats = np.linspace(0, 90, num=int(Number_Lats / 2) + 1, endpoint=True) * np.pi / 180
            A = np.abs(2 * np.pi * 6.371e6 ** 2 * np.diff(np.sin(Lats)))
            A = np.concatenate([A[::-1], A])
        else:
            Lats = np.linspace(90 / Number_Lats, 90, num=int((Number_Lats + 1) / 2), endpoint=True) * np.pi / 180
            A = np.abs(2 * np.pi * 6.371e6 ** 2 * np.diff(np.sin(Lats)))
            A = np.concatenate([A[::-1], A])
            A = np.insert(A, int(len(A) / 2), 2 * np.abs(2 * np.pi * 6.371e6 ** 2 * (np.sin(Lats[0]))))
    else:
        raise TypeError('Number_Lats must be a vector or an integer')

    if Area_or_Weights == 'Area':
        return A
    elif Area_or_Weights == 'Weights':
        return A / np.sum(A)
    else:
        raise ValueError('"Area" or "Weights" string specified incorrectly')

lats = np.flip(np.arange(0, 180, 2.5)-88.75)
T = np.flip(np.load('./../../../STONED Project/STONED 1.10/INPUT/Satellite_T_LAT.npy'))
S = np.flip(np.load('./../../../STONED Project/S_0BP.npy'))

"""
Plot of CERES Values
"""

A = 213.6
B = 1.75
D = 3
C = 6.5

years = 100

def Alpha(T):
    return 0.5 - 0.3*np.tanh((T-1.1)/22.2)

temp = np.zeros((years+1, 72))
temp[0,:] = np.copy(T)

for i in range(years):
    dT = S*(1-Alpha(temp[i,:])) - (A + B*temp[i,:]) + D*(np.average(temp[i,:], weights=Area_Calc(72)) - temp[i,:])
    temp[i+1,:] = temp[i,:] + dT/C
    if abs(np.average(temp[i,:]-temp[i-1,:], weights=Area_Calc(72))) < 0.005:
        print(i)
        break
print(np.average(temp[i,:], weights=Area_Calc(72)))

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10,7.5), gridspec_kw={'height_ratios': [2,1]})
ax1.scatter(lats, T, c='k', s=5, label='T$_0$')
ax1.scatter(lats, temp[i,:], c='r', s=5, label='T$_{Eq}$')
ax1.set_xlim((90,-90))
ax1.set_yticks(np.linspace(-90, 30, num = 9, endpoint = True))
ax1.set_xticks(np.linspace(90, -90, num = 5, endpoint = True))
ax1.set_ylim(-95, 30)
ax1.xaxis.set_minor_locator(MultipleLocator(5))
ax1.yaxis.set_minor_locator(MultipleLocator(5))
ax1.grid(True,  which='major', linestyle = '--', alpha=0.5)
ax1.set_xlabel('Latitude [degrees north]')
ax1.set_ylabel('Temperature ($^\circ$C)')
ax1.text(87, 15, r'(a)', color='k')
ax1.legend(markerscale=2., fancybox=False, edgecolor='k', prop={'size': 16})

ax2.scatter(np.arange(38), np.average(temp, axis=1, weights=Area_Calc(72))[:i], c='k', s=5)
ax2.set_yticks(np.linspace(-85, 15, num = 5, endpoint = True))
ax2.set_xticks(np.linspace(0, 40, num = 9, endpoint = True))
ax2.set_ylim(-85, 16)
ax2.xaxis.set_minor_locator(MultipleLocator(1))
ax2.yaxis.set_minor_locator(MultipleLocator(5))
ax2.set_xlim((-1,38))
ax2.grid(True,  which='major', linestyle = '--', alpha=0.5)
ax2.set_xlabel('Time (years)')
ax2.set_ylabel('Temperature ($^\circ$C)')
ax2.text(-0.3, -6, r'(b)', color='k')
fig.subplots_adjust(hspace=0.3)
plt.tight_layout()
plt.savefig('./CERES_Model_result.pdf', format='pdf', dpi=300)

"""
Plot of Norths 1975 Values
"""

A = 211.2
B = 1.55
D = 3.75
C = 6.5
years = 100


def Alpha(T):
    return np.where(T < -10, 0.62, 0.34 - 0.12*np.sin(lats * np.pi / 180)**2)

temp = np.zeros((years+1, 72))
temp[0,:] = np.copy(T)

for i in range(years):
    dT = S*(1-Alpha(temp[i,:])) - (A + B*temp[i,:]) + D*(np.average(temp[i,:], weights=Area_Calc(72)) - temp[i,:])
    temp[i+1,:] = temp[i,:] + dT/C
    if abs(np.average(temp[i,:]-temp[i-1,:], weights=Area_Calc(72))) < 0.005:
        print(i)
        break
print(np.average(temp[i,:], weights=Area_Calc(72)))
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10,7.5), gridspec_kw={'height_ratios': [2,1]})
ax1.scatter(lats, T, c='k', s=5, label='T$_0$')
ax1.scatter(lats, temp[i,:], c='r', s=5, label='T$_{Eq}$')
ax1.set_xlim((90,-90))
ax1.set_yticks(np.linspace(-45, 30, num = 6, endpoint = True))
ax1.set_xticks(np.linspace(90, -90, num = 5, endpoint = True))
ax1.set_ylim(-45, 30)
ax1.xaxis.set_minor_locator(MultipleLocator(5))
ax1.yaxis.set_minor_locator(MultipleLocator(5))
ax1.grid(True,  which='major', linestyle = '--', alpha=0.5)
ax1.set_xlabel('Latitude [degrees north]')
ax1.set_ylabel('Temperature ($^\circ$C)')
ax1.text(87, 21.5, r'(a)', color='k')
ax1.legend(markerscale=2., fancybox=False, edgecolor='k', prop={'size': 16})

ax2.scatter(np.arange(i), np.average(temp, axis=1, weights=Area_Calc(72))[:i], c='k', s=5)
ax2.set_yticks(np.linspace(13, 15, num = 3, endpoint = True).astype(int))
ax2.set_xticks(np.linspace(0, i-1, num = 7, endpoint = True))
ax2.set_ylim(13, 15)
ax2.xaxis.set_minor_locator(MultipleLocator(1))
ax2.yaxis.set_minor_locator(MultipleLocator(0.1))
ax2.set_xlim((-0.5,i))
ax2.grid(True,  which='major', linestyle = '--', alpha=0.5)
ax2.set_xlabel('Time (years)')
ax2.set_ylabel('Temperature ($^\circ$C)')
ax2.text(-0.3, 14.52, r'(b)', color='k')
fig.subplots_adjust(hspace=0.3)
plt.tight_layout()
plt.savefig('./North_1975_Model_result.pdf', format='pdf', dpi=300)

"""
Plot of McGuffie 1975 Values
"""

A = 204
B = 2.17
D = 3.81
C = 6.5
years = 100


def Alpha(T):
    return 0.5 - 0.2*np.tanh((T+8.8)/10)

temp = np.zeros((years+1, 72))
temp[0,:] = np.copy(T)

for i in range(years):
    dT = S*(1-Alpha(temp[i,:])) - (A + B*temp[i,:]) + D*(np.average(temp[i,:], weights=Area_Calc(72)) - temp[i,:])
    temp[i+1,:] = temp[i,:] + dT/C
    if abs(np.average(temp[i,:]-temp[i-1,:], weights=Area_Calc(72))) < 0.005:
        print(i)
        break
print(np.average(temp[i,:], weights=Area_Calc(72)))
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10,7.5), gridspec_kw={'height_ratios': [2,1]})
ax1.scatter(lats, T, c='k', s=5, label='T$_0$')
ax1.scatter(lats, temp[i,:], c='r', s=5, label='T$_{Eq}$')
ax1.set_xlim((90,-90))
ax1.set_yticks(np.linspace(-45, 30, num = 6, endpoint = True))
ax1.set_xticks(np.linspace(90, -90, num = 5, endpoint = True))
ax1.set_ylim(-45, 30)
ax1.xaxis.set_minor_locator(MultipleLocator(5))
ax1.yaxis.set_minor_locator(MultipleLocator(5))
ax1.grid(True,  which='major', linestyle = '--', alpha=0.5)
ax1.set_xlabel('Latitude [degrees north]')
ax1.set_ylabel('Temperature ($^\circ$C)')
ax1.text(87, 21.5, r'(a)', color='k')
ax1.legend(markerscale=2., fancybox=False, edgecolor='k', prop={'size': 16})

ax2.scatter(np.arange(i), np.average(temp, axis=1, weights=Area_Calc(72))[:i], c='k', s=5)
ax2.set_yticks(np.linspace(12, 15, num = 4, endpoint = True))
ax2.set_xticks(np.linspace(0, i-1, num = 9, endpoint = True))
ax2.set_ylim(12, 15)
ax2.xaxis.set_minor_locator(MultipleLocator(1))
ax2.yaxis.set_minor_locator(MultipleLocator(0.2))
ax2.set_xlim((-1,i))
ax2.grid(True,  which='major', linestyle = '--', alpha=0.5)
ax2.set_xlabel('Time (years)')
ax2.set_ylabel('Temperature ($^\circ$C)')
ax2.text(-0.65, 14.3, r'(b)', color='k')
fig.subplots_adjust(hspace=0.3)
plt.tight_layout()
plt.savefig('./McGuffie_Model_result.pdf', format='pdf', dpi=300)

"""
Plot of Equilibrium Values
"""

A = 202.8
B = 2.17
D = 3.81
C = 6.5
years = 100


def Alpha(T):
    return 0.5 - 0.2*np.tanh((T+8.8)/10)

temp = np.zeros((years+1, 72))
temp[0,:] = np.copy(T)

for i in range(years):
    dT = S*(1-Alpha(temp[i,:])) - (A + B*temp[i,:]) + D*(np.average(temp[i,:], weights=Area_Calc(72)) - temp[i,:])
    temp[i+1,:] = temp[i,:] + dT/C
    if abs(np.average(temp[i,:]-temp[i-1,:], weights=Area_Calc(72))) < 0.005:
        print(i)
        break
print(np.average(temp[i,:], weights=Area_Calc(72)))
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10,7.5), gridspec_kw={'height_ratios': [2,1]})
ax1.scatter(lats, T, c='k', s=5, label='T$_0$')
ax1.scatter(lats, temp[i,:], c='r', s=5, label='T$_{Eq}$')
ax1.set_xlim((90,-90))
ax1.set_yticks(np.linspace(-45, 30, num = 6, endpoint = True))
ax1.set_xticks(np.linspace(90, -90, num = 5, endpoint = True))
ax1.set_ylim(-45, 30)
ax1.xaxis.set_minor_locator(MultipleLocator(5))
ax1.yaxis.set_minor_locator(MultipleLocator(5))
ax1.grid(True,  which='major', linestyle = '--', alpha=0.5)
ax1.set_xlabel('Latitude [degrees north]')
ax1.set_ylabel('Temperature ($^\circ$C)')
ax1.text(87, 21.5, r'(a)', color='k')
ax1.legend(markerscale=2., fancybox=False, edgecolor='k', prop={'size': 16})

ax2.scatter(np.arange(i), np.average(temp, axis=1, weights=Area_Calc(72))[:i], c='k', s=5)
ax2.set_yticks(np.linspace(13, 15, num = 3, endpoint = True))
ax2.set_xticks(np.linspace(0, i-1, num = 12, endpoint = True))
ax2.set_ylim(13, 15)
ax2.xaxis.set_minor_locator(MultipleLocator(1))
ax2.yaxis.set_minor_locator(MultipleLocator(0.2))
ax2.set_xlim((-0.8,i))
ax2.grid(True,  which='major', linestyle = '--', alpha=0.5)
ax2.set_xlabel('Time (years)')
ax2.set_ylabel('Temperature ($^\circ$C)')
ax2.text(-0.4, 14.55, r'(b)', color='k')
fig.subplots_adjust(hspace=0.3)
plt.tight_layout()
plt.savefig('./Equilibrium_result.pdf', format='pdf', dpi=300)

"""
Plot of Equilibrium Values fpr ice line at 60 degrees North/South
"""
A = 202.8
B = 2.17
D = 3.81
C = 6.5
years = 100

def Alpha(T):
    return 0.5 - 0.2*np.tanh((T+8.8)/10)

temp = np.zeros((years+1, 72))
temp[0,:] = np.copy(T)

for i in range(years):
    dT = S*(1-Alpha(temp[i,:])) - (A + B*temp[i,:]) + D*(np.average(temp[i,:], weights=Area_Calc(72)) - temp[i,:])
    temp[i+1,:] = temp[i,:] + dT/C
    if abs(np.average(temp[i,:]-temp[i-1,:], weights=Area_Calc(72))) < 0.005:
        print(i)
        break
print(np.average(temp[i,:], weights=Area_Calc(72)))


fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10,7.5), gridspec_kw={'height_ratios': [2,1]})
#ax1.scatter(lats, T, c='k', s=5, label='T$_0$')
ax1.scatter(lats, temp[i,:], c='k', s=5, label='T$_{Eq}$')
ax1.set_xlim((90,-90))
ax1.set_yticks(np.linspace(-20, 30, num = 6, endpoint = True))
ax1.set_xticks(np.linspace(90, -90, num = 7, endpoint = True))
ax1.set_ylim(-20, 30)
ax1.xaxis.set_minor_locator(MultipleLocator(5))
ax1.yaxis.set_minor_locator(MultipleLocator(5))
ax1.grid(True,  which='major', linestyle = '--', alpha=0.5)
ax1.set_xlabel('Latitude [degrees north]')
ax1.set_ylabel('Temperature ($^\circ$C)')
ax1.text(87, 24, r'(a)', color='k')
ax1.vlines((-60,60), -45, 30, colors='k', linestyle='dashed', alpha=0.5)
#ax1.hlines(-10, -90, 90, colors='k', linestyle='dashed', alpha=0.5)
#ax1.legend(markerscale=2., fancybox=False, edgecolor='k', prop={'size': 16})

ax2.scatter(np.arange(i), np.average(temp, axis=1, weights=Area_Calc(72))[:i], c='k', s=5)
ax2.set_yticks(np.linspace(8, 16, num = 5, endpoint = True))
ax2.set_xticks(np.linspace(0, 60, num = 13, endpoint = True))
ax2.set_ylim(8, 16)
ax2.xaxis.set_minor_locator(MultipleLocator(1))
ax2.yaxis.set_minor_locator(MultipleLocator(1))
ax2.set_xlim((-0.8,55))
ax2.grid(True,  which='major', linestyle = '--', alpha=0.5)
ax2.set_xlabel('Time (years)')
ax2.set_ylabel('Temperature ($^\circ$C)')
ax2.text(0.2, 14.3, r'(b)', color='k')

temp = np.zeros((years+1, 72))
temp[0,:] = np.copy(T)

for i in range(years):
    dT = S*0.976*(1-Alpha(temp[i,:])) - (A + B*temp[i,:]) + D*(np.average(temp[i,:], weights=Area_Calc(72)) - temp[i,:])
    temp[i+1,:] = temp[i,:] + dT/C
    if abs(np.average(temp[i,:]-temp[i-1,:], weights=Area_Calc(72))) < 0.005:
        print(i)
        break
print(np.average(temp[i,:], weights=Area_Calc(72)))
ax1.scatter(lats, temp[i,:], c='b', s=5, label='T$_S$')

ax2.scatter(np.arange(i), np.average(temp, axis=1, weights=Area_Calc(72))[:i], c='b', s=5)

D = 3.25
temp = np.zeros((years+1, 72))
temp[0,:] = np.copy(T)

for i in range(years):
    dT = S*(1-Alpha(temp[i,:])) - (A + B*temp[i,:]) + D*(np.average(temp[i,:], weights=Area_Calc(72)) - temp[i,:])
    temp[i+1,:] = temp[i,:] + dT/C
    if abs(np.average(temp[i,:]-temp[i-1,:], weights=Area_Calc(72))) < 0.005:
        print(i)
        break
print(np.average(temp[i,:], weights=Area_Calc(72)))
ax1.scatter(lats, temp[i,:], c='r', s=5, label='T$_D$')
ax1.legend(markerscale=2., fancybox=False, edgecolor='k', prop={'size': 16})

ax2.scatter(np.arange(i), np.average(temp, axis=1, weights=Area_Calc(72))[:i], c='r', s=5)

fig.subplots_adjust(hspace=0.3)
plt.tight_layout()
plt.savefig('./Equilibrium_T_EQ_S_D.pdf', format='pdf', dpi=300)