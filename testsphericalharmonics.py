#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 10 09:58:01 2017

@author: monika
"""

import matplotlib.pyplot as plt
import numpy as np
from mathtools import sph_harm, sph_harm_diff_phi, sph_harm_diff_theta


m = 1
n = m - 0

theta = np.linspace(0, 2 * np.pi, 360)
phi   = np.linspace(0,     np.pi, 180)


x, y = np.meshgrid(theta, phi)

#sph_real = np.real(sph_harm_diff_phi(n, m, x, y))
#sph_imag = np.imag(sph_harm_diff_phi(n, m, x, y))
# LUB
sph_real = np.real(sph_harm(n, m, x, y))
sph_imag = np.imag(sph_harm(n, m, x, y))

fig = plt.figure(figsize=(8, 4))

ax1 = fig.add_subplot(121, polar=True)
ax1.yaxis.set(ticks=phi[::45], ticklabels=[0, 45, 90, 135])
ax1.tick_params('y', colors='red')

ax2 = fig.add_subplot(122, polar=True)
ax2.yaxis.set(ticks=phi[::45], ticklabels=[0, 45, 90, 135])
ax2.tick_params('y', colors='red')

pos1 = ax1.pcolormesh(x, y, sph_real, cmap='Greys')
pos2 = ax2.pcolormesh(x, y, sph_imag, cmap='Greys')

fig.colorbar(pos1, ax=ax1)
fig.colorbar(pos2, ax=ax2)

plt.show()
# =============================================================================
#
# sph_real = np.real(sph_harm(n, m, theta, phi))
# sph_imag = np.imag(sph_harm(n, m, theta, phi))
#
# fig = plt.figure(figsize=(8, 8))
# ax1 = fig.add_subplot(211)
# ax2 = fig.add_subplot(212)
# pos1 = ax1.imshow(sph_real)
# pos2 = ax2.imshow(sph_imag)
#
# fig.colorbar(pos1, ax=ax1)
# fig.colorbar(pos2, ax=ax2)
#
# plt.show()
#
# =============================================================================
