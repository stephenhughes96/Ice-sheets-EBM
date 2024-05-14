"""
The Energy balance model variance for each of the three cases T_Eq, T_D and T_s as presented in Chapter 5.4 Figure 5.14
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

lats = np.flip(np.arange(0, 180, 2.5) - 88.75)

R_2_5_T_Eq = np.load('R_2_5_T_Eq.npy')
R_2_5_T_S = np.load('R_2_5_T_S.npy')
R_2_5_T_D = np.load('R_2_5_T_D.npy')

P_2_5_T_Eq = np.load('P_2_5_T_Eq.npy')
P_2_5_T_S = np.load('P_2_5_T_S.npy')
P_2_5_T_D = np.load('P_2_5_T_D.npy')

R_5_T_Eq = np.load('R_5_T_Eq.npy')
R_5_T_S = np.load('R_5_T_S.npy')
R_5_T_D = np.load('R_5_T_D.npy')

P_5_T_Eq = np.load('P_5_T_Eq.npy')
P_5_T_S = np.load('P_5_T_S.npy')
P_5_T_D = np.load('P_5_T_D.npy')

fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, 1, figsize=(10,12))
fig.subplots_adjust(right=0.92, top=0.98, left=0.08, hspace=0, bottom=0.08)

ax1.scatter(lats, np.mean(R_2_5_T_Eq, axis=0), c='k', s=2, label='T$_{Eq}$')
ax1.scatter(lats, np.mean(R_2_5_T_S, axis=0), c='b', s=2, label='T$_{S}$')
ax1.scatter(lats, np.mean(R_2_5_T_D, axis=0), c='r', s=2, label='T$_{D}$')

ax1.set_xlim(90, -90)
ax1.set_ylim(0, 5)
ax1.set_ylabel(r'Variance (K$^2$)')
ax1.set_yticks(np.linspace(0, 4, num=3, endpoint=True))
ax1.yaxis.set_minor_locator(MultipleLocator(0.5))
ax1.xaxis.set_ticklabels([])
ax1.xaxis.set_ticks_position('none')
# ax1.title.set_visible(False)
ax1.set_xticks(np.linspace(90, -90, num=5, endpoint=True))
ax1.text(87, 4, r'(a)')
ax1.grid(True, which='major', linestyle='--', alpha=0.5)
ax1.legend(markerscale=2., fancybox=False, edgecolor='k', prop={'size': 16}, loc='upper center')

ax2.scatter(lats, np.mean(R_5_T_Eq, axis=0), c='k', s=2)
ax2.scatter(lats, np.mean(R_5_T_S, axis=0), c='b', s=2)
ax2.scatter(lats, np.mean(R_5_T_D, axis=0), c='r', s=2)

ax2.set_xlim(90, -90)
ax2.set_ylim(0,16.5)
ax2.set_yticks(np.linspace(0, 15, num=4, endpoint=True))
ax2.yaxis.set_label_position("right")
ax2.set_ylabel(r'Variance (K$^2$)')
ax2.yaxis.tick_right()
ax2.yaxis.set_minor_locator(MultipleLocator(1))
ax2.xaxis.set_ticklabels([])
ax2.xaxis.set_ticks_position('none')
# ax2.title.set_visible(False)
ax2.set_xticks(np.linspace(90, -90, num=5, endpoint=True))
ax2.text(87, 13, r'(b)')
ax2.grid(True, which='major', linestyle='--', alpha=0.5)

ax3.scatter(lats, np.mean(P_2_5_T_Eq, axis=0), c='k', s=2)
ax3.scatter(lats, np.mean(P_2_5_T_S, axis=0), c='b', s=2)
ax3.scatter(lats, np.mean(P_2_5_T_D, axis=0), c='r', s=2)

ax3.set_xlim(90, -90)
ax3.set_ylim(0, 10)
ax3.set_ylabel(r'Variance (K$^2$)')
ax3.set_yticks(np.linspace(0, 10, num=3, endpoint=True))
ax3.yaxis.set_minor_locator(MultipleLocator(0.5))
ax3.xaxis.set_ticklabels([])
ax3.xaxis.set_ticks_position('none')
# ax3.title.set_visible(False)
ax3.set_xticks(np.linspace(90, -90, num=5, endpoint=True))
ax3.text(87, 8, r'(c)')
ax3.grid(True, which='major', linestyle='--', alpha=0.5)

ax4.scatter(lats, np.mean(P_5_T_Eq, axis=0), c='k', s=2)
ax4.scatter(lats, np.mean(P_5_T_S, axis=0), c='b', s=2)
ax4.scatter(lats, np.mean(P_5_T_D, axis=0), c='r', s=2)

ax4.set_xlim(90, -90)
ax4.set_ylim(0, 55)
ax4.set_yticks(np.linspace(0, 50, num=3, endpoint=True))
ax4.yaxis.set_label_position("right")
ax4.set_ylabel(r'Variance (K$^2$)')
ax4.yaxis.tick_right()
ax4.yaxis.set_minor_locator(MultipleLocator(5))
ax4.xaxis.set_ticklabels([])
ax4.xaxis.set_ticks_position('none')
ax4.set_xticks(np.linspace(90, -90, num=5, endpoint=True))
ax4.text(87, 44, r'(d)')
ax4.grid(True, which='major', linestyle='--', alpha=0.5)

P_5_T_Eq_Ice_List = []
P_5_T_S_Ice_List = []
P_5_T_D_Ice_List = []

for i in range(10000):
    if P_5_T_Eq[i, 35] < 7.5:
        P_5_T_Eq_Ice_List.append(P_5_T_Eq[i, :])
    if P_5_T_S[i, 35] < 7.5:
        P_5_T_S_Ice_List.append(P_5_T_S[i, :])
    if P_5_T_D[i, 35] < 7.5:
        P_5_T_D_Ice_List.append(P_5_T_D[i, :])

P_5_T_Eq_Ice= np.array(P_5_T_Eq_Ice_List)
P_5_T_S_Ice = np.array(P_5_T_S_Ice_List)
P_5_T_D_Ice =np.array(P_5_T_D_Ice_List)

ax5.scatter(lats, np.mean(P_5_T_Eq_Ice, axis=0), c='k', s=2, label='T$_{Eq}$')
ax5.scatter(lats, np.mean(P_5_T_S_Ice, axis=0), c='b', s=2, label='T$_{S}$')
ax5.scatter(lats, np.mean(P_5_T_D_Ice, axis=0), c='r', s=2, label='T$_{D}$')

ax5.set_xlim(90, -90)
ax5.set_ylim(0, 30)
ax5.set_ylabel(r'Variance (K$^2$)')
ax5.set_yticks(np.linspace(0, 30, num=4, endpoint=True))
ax5.yaxis.set_minor_locator(MultipleLocator(5))

# ax3.title.set_visible(False)
ax5.xaxis.set_minor_locator(MultipleLocator(5))
ax5.set_xticks(np.linspace(90, -90, num=5, endpoint=True))
ax5.set_xlabel('Latitude [degrees north]')
ax5.text(87, 24, r'(e)')
ax5.grid(True, which='major', linestyle='--', alpha=0.5)

plt.savefig('./Latitudinal_Variance_All.pdf', format='pdf', dpi=300)