# -*- coding: utf-8 -*-
"""
Created on Thu Mar 22 14:23:37 2018

@author: lampio
"""
import scipy as sp
from matplotlib import pyplot as plt
from scipy import linalg
from scipy.sparse import coo_matrix
from scipy.sparse import linalg
from seaborn import heatmap

def phi_avg_from_Sun_at_Tampere(n):
    # Gives average q_sun per day
    # n is day between 0-364
    # m is month between -0.5 - 11.5
    m = -0.5 + n/365*12
    phi_avg = 181.5*sp.sin(0.2683*m + 0.221) + 52.64*sp.sin(0.8171*m + 3.764)
    if phi_avg < 3.0: # Fit was not perfect for all winter days, a fix
        phi_avg = 3.0
    return phi_avg

def T_avg_at_Tampere(n):
    # Gives average T_air per day
    # n is day between 0-364
    # m is month between -0.5 - 11.5
    m = -0.5 + n/365*12
    T_avg = 103.9*sp.sin(0.2112*m - 1.029) + 110.5*sp.sin(0.1832*m + 2.188) + 8.141*sp.sin(0.6231*m + 4.37)
    return T_avg

def T_sky_avg(T_avg):
    return T_avg - 25.0

def plot_results(T, timestep):
    # Printing the results
    # Example figure, you must refine this better to show dimensions etc.!
    plt.figure()
    plt.contourf(T[:,:,timestep], 20, cmap='jet') # Temperature distribution at last time step
    plt.colorbar()

def energy_pile(r0, R, Z, H, dt, N_timesteps, M, N, k, rho, cp, h_air, h_rad_sky, U_ep, T_ep, T0, heating_starts, heating_ends, print_gap = 100, plot_gap = 0):
    
    U_ep_memory = float(U_ep) # Store the value for memory
        
    T = sp.ones((M,N,N_timesteps))*T0 # Initialization of temperatures
    TT_old = sp.ones((1,N*M))*T0 # Initialization of old temperatures
    TT_old = TT_old[0]
    dr = R/M # Discretization in r-direction
    dz = Z/N # Discretization in z-direction
    dz_ep = int(H/dz) # Cell number for energy pile bottom
    r = sp.linspace(r0, R, M+1) # Boundary locations in r-direction
    A_r = 2*sp.pi*r*dz # Surface areas in r-direction
    A_u = sp.pi*(r[1:]**2 - r[0:-1]**2) # Surface areas in z-direction
    V = A_u*dz # Cell volumes
    m = V*rho # Cell masses
    
    # Constant part of matrix
    # Coefficients
    C1 = sp.zeros((M,N))
    C2 = sp.zeros((M,N))
    Cu = sp.zeros((M,N))
    Cd = sp.zeros((M,N))
    Cp = sp.zeros((M,N))
    for i in range(M):
        for j in range(N):
            C1[i,j] = k*A_r[i]/dr/m[i]/cp
            C2[i,j] = k*A_r[i+1]/dr/m[i]/cp

            Cu[i,j] = k*A_u[i]/dz/m[i]/cp
            Cd[i,j] = k*A_u[i]/dz/m[i]/cp

    # insulation for conduction at r = R
    C2[-1,:] = 0
    
    # insulation for conduction at r = r0
    C1[0,:] = 0
  
    # insulation for conduction at r = r0
    Cu[:,0] = 0
    Cd[:,-1] = 0

    # Constant coefficients for diagonal
    for i in range(M):
        for j in range(N):
            Cp[i,j] = -(C1[i,j] + C2[i,j] + Cu[i,j] + Cd[i,j] + 1.0/dt)

    # Calculating indexes ans coefficients for sparse matrix
    II = [] # index i
    JJ = [] # index j
    DATA = [] # value
    for i in range(M):
        for j in range(N):
            ii = i + j*M
            II.append(ii)
            JJ.append(ii)
            DATA.append(Cp[i,j])
            
            if ii < N*M-1:
                II.append(ii)
                JJ.append(ii+1)
                DATA.append(C2[i,j])
            
            if ii > 0:
                II.append(ii)
                JJ.append(ii-1)
                DATA.append(C1[i,j])
            
            if ii < N*M - M:
                II.append(ii)
                JJ.append(ii+M)
                DATA.append(Cd[i,j])
            
            if ii > M-1:
                II.append(ii)
                JJ.append(ii-M)
                DATA.append(Cu[i,j])

    # Building a sparse matrix (COO-type) for constant coefficients
    A_const = coo_matrix((DATA, (II, JJ)), shape=(N*M, N*M))
    
    # Iteration in time (backward Euler method)
    for time in range(N_timesteps-1):
        current_day = time*dt/3600/24 # Day number
        
        # Here you have to figure out how to set energy pile OFF
        # for first year
        if current_day > 606 and (sp.mod(current_day, 365) < heating_ends or sp.mod(current_day, 365) > heating_starts):
            U_ep = U_ep_memory
        else:
            U_ep = 0.0
            
        T_air = T_avg_at_Tampere(sp.mod(current_day, 365))
        T_sky = T_sky_avg(T_air)
        phi_avg = phi_avg_from_Sun_at_Tampere(sp.mod(current_day, 365))
        if print_gap != 0:
            if sp.mod(time, print_gap) == 0:
                print("Iteration round:", time, 
                      "day:", '{:6.2f}'.format(current_day), 
                      "T_ambient:", '{:6.2f}'.format(T_air), 
                      "Solar irradiation:", '{:6.2f}'.format(phi_avg))
        
        b = sp.zeros((N*M,1)) # Source term vector
        II = [] # i - index
        JJ = [] # j - index
        DATA = [] # value
        for i in range(M):
            # convection h_air at z = 0
            ii = i + 0*M
            II.append(ii)
            JJ.append(ii)
            DATA.append(-h_air*A_u[i]/m[i]/cp)
            b[ii] = b[ii] - h_air*A_u[i]*T_air/m[i]/cp
            
            # Sun irradiation at z = 0
            b[ii] = b[ii] - phi_avg*A_u[i]/m[i]/cp
            
            # Radiation exchange with sky at z = 0
            II.append(ii)
            JJ.append(ii)
            DATA.append(-h_rad_sky*A_u[i]/m[i]/cp)
            b[ii] = b[ii] - h_rad_sky*A_u[i]/m[i]/cp*T_sky
            
            
        for j in range(N):
            # Energy pile at r = r0
            ii = 0 + j*M
            if j < dz_ep:
                II.append(ii)
                JJ.append(ii)
                DATA.append(-U_ep*A_r[0]/m[0]/cp)
                b[ii] = b[ii] - U_ep*A_r[0]*T_ep/m[0]/cp

        # Effect of last time step temperatures
        b = b - TT_old.reshape([N*M,1])/dt

        # Sparse matrix for changing coefficients
        A_new = coo_matrix((DATA, (II, JJ)), shape=(N*M, N*M))
        
        # Solving linear system by summing constant and changing part
        TT = sp.sparse.linalg.spsolve((A_const + A_new).tocsc(), b)
        
        # Forming a matrix T from TT vector
        for i in range(M):
            for j in range(N):
                ii = i + j*M
                T[i,j,time+1] = TT[ii]
        
        # Calculate that the overall heat balance holds for domain

        TT_old = TT.astype(float)
        
#        print(T)
        
        if plot_gap != 0:
            if sp.mod(time, plot_gap) == 0:
                print("Iteration round:", time, 
                      "day:", '{:6.2f}'.format(current_day))
                plot_results(T, time)

    return T


# Calculations

# Fill correct values to "<you must determine>" spaces

# Dimensions
r0 = 0.6 # Radius of energy pile m
R = 10 # Outer radius of calculation domain m
Z = 30 # Height of domain m
H = 15 + 4 # Height of energy pile m

# Time stepping details
dt = 60*60 # Time step length s
N_days = 2*365
N_timesteps = int(N_days*3600*24/dt) # Number of time steps
print("Total timesteps:",N_timesteps)

# Discretization
# Tip: try out first with coarse meshes and refine just for actual calculations
# Caution: In this code H/(Z/N) must be an integer!
# Otherwise: Some leftover height of energy pile is not included in calculations
M = R*2 # Number of control volumes in r - direction
N = Z*2 # Number of control volumes in z - direction 

# Thermal properties of soil
k = 1.1 #W/mK
rho = 1750 # kg/m3
cp = 1380 # J/kgK

# Boundary conditions
h_air = 10 # Average convective heat transfer coefficient W/m2K
h_rad_sky = 4*5.67e-8*40**3 # Average radiation heat transfer coefficient

# U-value of energy pile
r1 = 0.1
r2 = r0
l = 0.3
S = 2*sp.pi*H/sp.arccosh((r1**2+r2**2-l**2)/(2*r1*r2))
print("Shape factor:",S)
U_ep = 10 # U-value of energy pile

# Temperatures
T_ep = -6 + 4/3 # Fluid temperature C
T0 = 5 # Initial temperature of soil

# Heating days per one year (days 0-364)
heating_starts = 31+28+31+30+31+30+31+31 # September 1st
heating_ends = 31+28+31+29 # 30th April 

# Printing of iteration rounds
# 0 for no printing/plotting
print_gap = 100 # You can variate this
plot_gap = 0

# Running the code
T = energy_pile(r0, R, Z, H, dt, N_timesteps, M, N, k, rho, cp, h_air, h_rad_sky, U_ep, T_ep, T0, heating_starts, heating_ends, print_gap, plot_gap)

plot_results(T, -1)

T1 = sp.transpose(T, (1,0,2))

plt.figure(figsize=(12, 8))
heatmap(T1[:,:,-1])
plt.xticks(sp.arange(1, 20, 2), sp.arange(1, 11))
plt.yticks(sp.arange(0, 60, 10), sp.arange(0, 31, 5))