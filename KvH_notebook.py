# -*- coding: utf-8 -*-
"""
Created on Tue May 10 12:38:28 2022

@author: Chris
"""

import numpy as np
import matplotlib.pyplot as plt
import CEF_mag_calc as CEF

J = 7/2
kB  = 8.617e-2  # [meV/K];

p0 = np.array([1,  -0.4233,   0.0117,   0.5494,   0.00035,   0.0052,   -0.00045])

Jxx, Jzz = 0.5388, 0.6145
exch_XXZ = kB*np.diag([Jxx,Jxx,Jzz]) 

model = CEF.model_CEF_HighSymm(p0,exch_XXZ)

N = 50;
hv = np.linspace(0,60,N)

H = np.zeros((N,3))
M = np.zeros((N,3))

H[:,0] = hv;

plt.figure()

plt.subplot(1,2,1)
for t in [4,12,20,30,50,70]:
    k = model.compute_kabvH(hv,t)
    plt.plot(hv,k)
    
plt.subplot(1,2,2)
for t in [4,12,20,30,50,70]:
    k = model.compute_kcvH(hv,t)
    plt.plot(hv,k)   

plt.show()


