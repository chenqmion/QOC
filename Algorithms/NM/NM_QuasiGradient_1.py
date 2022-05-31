import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 6})
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.gridspec as gridspec

from qutip import *
import random

import Task_1qubit_1cavity_dispersive_parametrized as tk

#%% read parameters
T = tk.T 
tau = tk.tau 
t_list = tk.t_list

Hd = tk.H_d
Hc = tk.H_c
Nc = len(Hc)

xi = tk.xi 
N_params = 3* 3

#%%
eta = tk.eta

def DRAG_Gaussian(xi, sigm=T/2):
    u_x = np.exp(-(t_list-T/2)**2/(2*sigm**2))
    u_x = u_x - u_x[0]
    
    xi_DRAG = np.pi/2/(np.sum(u_x)*tau)
    u_x = xi_DRAG * u_x
    
    u_y = (t_list-T/2)/(sigm**2) * u_x/eta
    return np.array([u_x,u_y])

u_DRAG = DRAG_Gaussian(xi)

#%% PSO
InFid_goal = 1E-4
N_rep = 1 # run algorithm N_rep times
N_samp = Nc*N_params + 1
N_gen = int(1E7)

c1 = 1.
c2 = 2.
c3 = 0.5
c4 = 0.5
c5 = 0.1

F_rep = []
u_rep = []
for num_rep in range(0,N_rep): 
    # initial guess
    u_pre = np.min(xi/N_params) * (2*np.random.rand(N_samp,Nc,N_params)-1)
    # u_pre[0] = np.load('u_best.npy')
    
    # u_pre = (1 + 0.01 * (2*np.random.rand(N_samp,Nc,M)-1))
    # u_pre = u_pre * u_DRAG
    
    Fid_pre = []
    for num_samp in range(0,N_samp):
        Fid_pre.append(tk.Cal_InFid1(u_pre[num_samp]))
    Fid_pre = np.array(Fid_pre).ravel()
    F_sequence = np.argsort(Fid_pre)
    Fid_pre = Fid_pre[F_sequence]
    u_pre = u_pre[F_sequence]
    
    u_best = u_pre[0]
    F_best = [Fid_pre[0]]

    for num_gen in range(0,N_gen):
        # center
        x_m = np.mean(u_pre[:-1], axis=0)
        # reflection
        x_r = x_m + c1 * (x_m-u_pre[-1])
        F_r = tk.Cal_InFid1(x_r)
        
        # quasi gradient
        if (F_r > Fid_pre[0]):
            x_s = np.array([u_pre[i].ravel()[i] for i in range(0,Nc*N_params)]).reshape(Nc,N_params)
            F_s = tk.Cal_InFid1(x_s)
            
            g = np.zeros((Nc,N_params))
            for num_g in range(0,Nc*N_params,2):
                g.ravel()[num_g] = (Fid_pre[num_g+1]-F_s)/(u_pre[num_g+1].ravel()[num_g]-x_s.ravel()[num_g])
            for num_g in range(1,Nc*N_params,2):
                g.ravel()[num_g] = (Fid_pre[num_g-1]-F_s)/(u_pre[num_g-1].ravel()[num_g]-x_s.ravel()[num_g])
                        
            if np.isnan(np.linalg.norm(g)):
                ind_nan = np.isnan(g.ravel())
                g.ravel()[ind_nan] = np.random.rand(len(ind_nan))
               
            g = g/np.linalg.norm(g)
            x_m2 = np.copy(u_pre[0])
            x_r2 = x_m2 - c1 * g * np.abs(np.sum(g*(u_pre[0]-u_pre[-1])))
            F_r2 = tk.Cal_InFid1(x_r2)
            if (F_r2 < F_r):
                x_m = x_m2
                x_r = x_r2
                F_r = F_r2
                
        #%% standard NM
        if (F_r >= Fid_pre[0]) and (F_r < Fid_pre[-2]):
            Fid_pre[-1] = F_r
            u_pre[-1] = x_r
        elif (F_r < Fid_pre[0]):
            # expansion
            x_e = x_m + c2 * (x_r-x_m)
            F_e = tk.Cal_InFid1(x_e)
            if (F_e < Fid_pre[0]):
                Fid_pre[-1] = F_e
                u_pre[-1] = x_e
            else:
                Fid_pre[-1] = F_r
                u_pre[-1] = x_r
        elif (F_r < Fid_pre[-1]):
            # outer contraction
            x_c = x_m + c3 * (x_r-x_m)
            F_c = tk.Cal_InFid1(x_c)
            if (F_c < Fid_pre[-1]):
                Fid_pre[-1] = F_c
                u_pre[-1] = x_c
            else:
               # shrink
               for num_samp in range(1,N_samp):
                   u_pre[num_samp] = u_pre[0] + c4 * (u_pre[num_samp]-u_pre[0])
                   Fid_pre[num_samp] = tk.Cal_InFid1(u_pre[num_samp])
        else:
            # inner contraction
            x_c = x_m - c3 * (x_r-x_m)
            F_c = tk.Cal_InFid1(x_c)
            if (F_c < Fid_pre[-1]):
                Fid_pre[-1] = F_c
                u_pre[-1] = x_c
            else:
                # shrink
                for num_samp in range(1,N_samp):
                    u_pre[num_samp] = u_pre[0] + c4 * (u_pre[num_samp]-u_pre[0])
                    Fid_pre[num_samp] = tk.Cal_InFid1(u_pre[num_samp])

        # perturbation
        if (num_gen%1E4 == int(1E4-1)) and ((F_best[-1000]-F_best[-1]) < F_best[-1]/1E3):
            print('perturb')
            if (np.random.rand() < 0.1):
                u_pre[0] = u_pre[0] * (1 + F_best[-1]*(2*np.random.rand(Nc,N_params)-1))
                print('adjust best')
                
            u_pre[1:] = np.min(xi/100)*(2*np.random.rand(N_samp-1,Nc,N_params)-1) 
            for num_samp in range(0,N_samp):
                Fid_pre[num_samp] = tk.Cal_InFid1(u_pre[num_samp])
        
        # constrain
        bd = (np.ones((N_samp,N_params,Nc)) * xi).transpose()
        bd[:,:int(N_params/3),:] = 0.5
        # bd[:,:int(N_params/3),:] = 0.5 * (2*np.pi)
        
        ind_bd = np.where(np.abs(u_pre.ravel()) > bd.ravel())
        u_pre.ravel()[ind_bd] = (2*np.random.rand(len(ind_bd))-1) * bd.ravel()[ind_bd]
        samp_list = np.divmod(ind_bd,N_params*Nc)[0].ravel()
        for ind_samp in np.unique(samp_list):
            Fid_pre[ind_samp] = tk.Cal_InFid1(u_pre[ind_samp])
        
        # sort
        F_sequence = np.argsort(Fid_pre)
        Fid_pre = Fid_pre[F_sequence]
        u_pre = u_pre[F_sequence]
    
        # update best
        u_best = u_pre[0]
        F_best.append(Fid_pre[0])
        if (num_gen%10==0):
            print('NM:', num_rep, num_gen, F_best[-1], np.mean(Fid_pre))
       
        if F_best[-1] < InFid_goal:
            print('NM:', num_rep, num_gen, F_best[-1], np.mean(Fid_pre))
            break

    F_rep.append(F_best)
    u_rep.append(u_best)
            
    np.save('F_rep.npy',F_rep)
    np.save('u_rep.npy',u_rep)

np.save('F_rep.npy',F_rep)
np.save('u_rep.npy',u_rep)
    

#%% plots
F_rep = np.load('F_rep.npy', allow_pickle=True)
u_rep = np.load('u_rep.npy', allow_pickle=True)
# u_t = tk.PulseGenerator(u_rep[-1])
u_t = tk.PulseGenerator(u_best)

fig = plt.figure(figsize=(3.5,3))
gr = plt.GridSpec(2,1, wspace=0.0, hspace=0.5)
ax1 = plt.subplot(gr[0,0])
ax2 = plt.subplot(gr[1,0], yscale='log')
for m in range(0,Nc):
    if (m==0):
        ax1.plot(t_list, u_t[m]*1E3/(2*np.pi), color='blue', linewidth=1., label='x')
    else:
        ax1.plot(t_list, u_t[m]*1E3/(2*np.pi), color='red', linewidth=1., label='y')

ax1.legend()
ax2.plot(F_rep[-1], color='red', linewidth=1.)
       
ax1.set_xlim(0,T)
ax2.set_ylim(5e-5,2)

ax1.set_xlabel('t (ns)')
ax2.set_xlabel('Iteration')

ax1.set_ylabel(r'$u_k(t)$ ($\rm 2\pi\times MHz$)')
ax2.set_ylabel(r'$1-J(t)$')
    
fig.align_ylabels()

# plt.savefig('Fig_.pdf', bbox_inches='tight') 
plt.show()
