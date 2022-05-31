import numpy as np
import matplotlib.pyplot as plt
from qutip import *

import scipy.linalg as lin
import time as time
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

#%% controller
g = 0.1 * (2*np.pi) 
eta = -0.3 * (2*np.pi) 
xi = 0.1 * (2*np.pi) 

w_q = 5 * (2*np.pi) 
w_a = 7 * (2*np.pi) 

# rotating frame
w_sub = w_q - w_a
w_q_disp = (w_q+w_a)/2 - np.sqrt(w_sub**2 + 4*g**2)/2

w_d = w_q
# w_d = w_a
w_q = w_q - w_d
w_a = w_a - w_d

# time
T = 10. # ns
M = int(1e2)
tau = T/M
t_list = np.linspace(tau/2,T-tau/2,M)

#%% single qubit operator
N_q = 3
I_q = qeye(N_q)
sigma_p = create(N_q)
sigma_m = destroy(N_q)

sigma_x = sigma_p + sigma_m
sigma_y = -1j*( sigma_p - sigma_m )
sigma_z = sigma_p * sigma_m

#%% resonator
N_a = 3
I_a = qeye(N_a)
a_p = create(N_a)
a_m = destroy(N_a)

#%% Hamiltonian
H_q = w_q * tensor(sigma_p * sigma_m, I_a)
H_q2 = (eta/2) * tensor(sigma_p * sigma_p * sigma_m * sigma_m, I_a)
H_a = w_a * tensor(I_q, a_p * a_m)
H_aq = g * tensor(sigma_x, (a_p+a_m))

H_x = tensor(sigma_x, I_a)
H_y = tensor(sigma_y, I_a)

H_d = (H_q + H_q2) + H_a + H_aq
H_c = [H_x, H_y]

#%% target
ket_0 = fock(N_a,0)
ket_1 = fock(N_a,1)
ket_g = fock(N_q,0)
ket_e = fock(N_q,1)
ket_f = fock(N_q,2)

ket_ini = tensor(ket_g, ket_0) # 0g
# ket_fin = tensor(ket_e, ket_0) #  -> 0e
ket_fin = tensor(ket_f, ket_0) # -> 0f
# ket_fin = tensor(ket_g, ket_1) # -> 1g

def Cal_InFid1(u, H_d=H_d, H_c=H_c, param=True):
    if param == True:
        u_t = PulseGenerator(u)
    else:
        u_t = u
        
    H = [H_d]
    for num_k in range(0,len(u_t)):
        H.append([H_c[num_k],u_t[num_k]])
    
    res_t = mesolve(H, ket_ini, t_list, c_ops=[])
    Fid = (fidelity(ket_fin, res_t.states[-1]))**2
    return 1 - Fid

#%% controller
def PulseGenerator(u):
    N_params = len(u[0])
    r_list = u[:, :int(N_params/3)]
    a_list = u[:, int(N_params/3):int(2*N_params/3)]
    b_list = u[:, int(2*N_params/3):]
    
    nu_list = (2*np.pi)/T * (range(len(r_list[0]))+r_list)
    u_sin = np.sin(np.einsum('ij,k->ijk', nu_list, t_list))
    # u_t = np.einsum('ij,ijk->ik', a_list, u_sin)
    
    u_cos = np.cos(np.einsum('ij,k->ijk', nu_list, t_list))
    u_t = np.einsum('ij,ijk->ik', a_list, u_sin) + np.einsum('ij,ijk->ik', b_list, u_cos)
    return u_t

# def PulseGenerator(u):
#     N_params = len(u[0])
#     r_list = u[:, :int(N_params/3)]
#     a_list = u[:, int(N_params/3):int(2*N_params/3)]
#     b_list = u[:, int(2*N_params/3):]
    
#     u_sin = np.sin(np.einsum('ij,k->ijk', r_list, t_list))
#     u_cos = np.cos(np.einsum('ij,k->ijk', r_list, t_list))
#     u_t = np.einsum('ij,ijk->ik', a_list, u_sin) + np.einsum('ij,ijk->ik', b_list, u_cos)
#     return u_t

# #%% crosscheck
# def Cal_InFid2(um):
#     Ut, U = Ref_TimeEvolution2(H_d, H_c, um, tau)
#     Fid = np.abs(ket_fin.full().ravel() @ Ut @ ket_ini.full().ravel())**2
#     return 1 - Fid

# def Ref_PixelEvolution2(H0, Hc, um, tau):
#     H = H0.full()
#     for num_1 in range(0, len(um)):
#         H = H + um[num_1]*Hc[num_1].full()
#     Um = lin.expm( -1j*tau*H )
#     return Um

# def Ref_TimeEvolution2(H0, Hc, u, tau, param=True):
#     if param == True:
#         u_t = PulseGenerator(u)
#     else:
#         u_t = u

#     U = Ref_PixelEvolution2(H0, Hc, np.array(u_t)[:,0], tau)
#     U_list = [U]

#     for num_m in range(1,len(np.array(u_t)[0])):
#         Um = Ref_PixelEvolution2(H0, Hc, np.array(u_t)[:,num_m], tau)
#         U = Um @ U
#         U_list.append(Um)

#     U_out = [U, U_list]  
#     return U_out