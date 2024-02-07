#!/usr/bin/env python3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import csv
import math 
import matplotlib.image as img
from matplotlib import interactive
import os 
import random

# Load CSV file into a Pandas DataFrame
df = pd.read_csv("averages.csv")

# Extract relevant columns
X = df['X'].values
Y = df['Y'].values
Z = df['Z'].values

u_mean = df['avgProfile'].values
uprime2mean = df[['avgProfile_uv', 'avgProfile_uw']].values
nut = df['avgProfile_nut'].values

uv = uprime2mean[:, 0]
uw = uprime2mean[:, 1]

nu = 8.25 * 10**(-5)
Ustar = np.sqrt((0.05 * 9.81) / 2)

def coord(X, Y, Z, uv, uw):
    x = Y
    y = Z
    z = X    
    r = np.sqrt(x*x + y*y)
    theta = np.arctan2(y, x)
    uw_prime = uv * np.cos(theta) + uw * np.sin(theta)
    return r, uw_prime

def totalStress(X, Y, Z, UMean, uv, uw, nut):
    x = Y
    y = Z
    z = X
    r = np.sqrt(x * x + y * y)what 
    theta = np.arctan2(y, x)

    dr = np.zeros_like(r)
    rm = 0.5 * (r[:-1] + r[1:])  # linear interpolant of r at midpoints
    
    dr[1:-1] = rm[1:] - rm[:-1]  # cell sizes for symmetric two-sided difference in bulk
    dr[0] = r[1] - r[0]     # cell size for one-sided difference at inner wall
    dr[-1] = r[-1] - r[-2]  # cell size for one-sided difference at outer wall

    dU = np.zeros_like(UMean)
    Um = 0.5 * (UMean[:-1] + UMean[1:])  # linear interpolant of U at midpoints
    dU[1:-1] = Um[1:] - Um[:-1]  # symmetric two-sided finite difference stencil

    dU[0] = UMean[1] - UMean[0]     # one-sided difference at inner wall
    dU[-1] = UMean[-1] - UMean[-2]  # one-sided difference at outer wall
    uw_prime = uv * np.cos(theta) + uw * np.sin(theta)  # Reynolds Stress RZ

    Tau_prime = -uw_prime + nut * dU / dr
    Tau = Tau_prime + nu * dU / dr

    iMax = np.argmax(UMean)

    r_i = r[0:iMax]
    r_o = np.flipud(r[iMax+1:])

    UMean_i = UMean[0:iMax]
    UMean_o = np.flipud(UMean[iMax+1:])

    tauW_i = uw_prime[0] - (nu + nut[0]) * (UMean[1] - UMean[0]) / (r[1] - r[0])   # FDM at inner wall
    tauW_o = uw_prime[-1] - (nu + nut[-1]) * (UMean[-1] - UMean[-2]) / (r[-1] - r[-2])   # FDM at outer wall

    uTau_i = np.sqrt(np.abs(tauW_i))
    uTau_o = np.sqrt(np.abs(tauW_o))

    deltaNu_i = nu / uTau_i
    deltaNu_o = nu / uTau_o

    rPlus_i = np.abs(r_i - r[0]) / deltaNu_i
    rPlus_o = np.abs(r_o - r[-1]) / deltaNu_o

    return r, Tau, Tau_prime, rPlus_i, rPlus_o, UMean_i, UMean_o


r = coord(X, Y, Z, uv, uw)[0]
uw_stress = coord(X, Y, Z, uv, uw)[1]


eta = totalStress(X,Y,Z,u_mean,uv,uw,nut)

r1		    = eta[0]
Tau1	    = eta[1]
Tau_prime1	= eta[2]
rPlusi	    = eta[3]
rPluso	    = eta[4]
uPlusi	    = eta[5]/ Ustar
uPluso	    = eta[6]/ Ustar


plt.figure(1)
plt.plot(r / (2 * 0.05), u_mean / (Ustar), 'r*-')
plt.xlabel("r/R")
plt.ylabel("u(r)/U*")
plt.title("Mean velocity Profile")
plt.grid(True)
plt.show()

plt.figure(2)
plt.plot(r / (2 * 0.05), uprime2mean[:, 0] / (Ustar * Ustar), 'r*-')
plt.plot(r / (2 * 0.05), uprime2mean[:, 1] / (Ustar * Ustar), 'b*-')
plt.xlabel("r/2R")
plt.ylabel("uv-uw/Ustar^2")
plt.title("Reynolds stresses")
plt.grid(True)
plt.show()

plt.figure(3)
plt.plot(r / (2 * 0.05), nut / (Ustar), 'r*-')
plt.xlabel("r/R")
plt.ylabel("u(r)/U*")
plt.title("nut")
plt.grid(True)
plt.show()

plt.figure(4)
plt.plot(r / (2 * 0.05), uw_stress / (Ustar * Ustar), 'r*-')
plt.xlabel("r/R")
plt.ylabel("umea/Ustar^2")
plt.title("ReynoldsStressAvg")
plt.grid(True)
plt.show()

plt.figure(5)
plt.plot(r / (2 * 0.05), Tau1 / (Ustar * Ustar), 'r*-')
plt.xlabel("r/R")
plt.ylabel("Tau/Ustar^2")
plt.title("Totalstress")
plt.grid(True)
plt.show()

plt.figure(6)
plt.plot(r / (2 * 0.05), Tau_prime1 / (Ustar * Ustar), 'r*-')
plt.xlabel("r/R")
plt.ylabel("TauPrime/Ustar^2")
plt.title("Tau prime")
plt.grid(True)
plt.show()

rPlusi = np.delete(rPlusi, [0])
uPlusi = np.delete(uPlusi, [0])
rPluso = np.delete(rPluso, [0])
uPluso = np.delete(uPluso, [0])

plt.figure(7)
plt.plot(rPlusi, uPlusi, 'r--')
plt.plot(rPluso, uPluso, 'k--')
plt.xlabel("r/R")
plt.ylabel("uplus inner/Ustar")
plt.xscale("log")
plt.title("uplusi")
plt.grid(True)
plt.show()

plt.figure(8)
plt.plot(rPluso, uPluso, 'r--')
plt.xlabel("r/R")
plt.ylabel("uplus outer/Ustar")
plt.xscale("log")
# plt.xlim(1,150)
plt.title("upluso")
plt.grid(True)
plt.show()