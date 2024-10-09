# noinspection PyInterpreter
import sys
from time import sleep
import warnings

import pylab as p
import scipy.optimize

warnings.filterwarnings('ignore')
warnings.warn('DelftStack')
warnings.warn('Do not show this message')
# print("No Warning Shown")

import torch
import torch.optim as optim
import torch.nn as nn
import copy

import numpy as np
import numpy as npp
import math
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.integrate import solve_ivp
from scipy.optimize import fixed_point
from itertools import product
from itertools import combinations_with_replacement
import statistics
import sympy as sp

import autograd.numpy as np
from autograd import grad
from autograd import jacobian
from autograd import hessian
from torchdiffeq import odeint

import time
import datetime
from datetime import datetime as dtime

import webcolors



dyn_syst = "I_Pendulum"         # Dynamical system studied (choice between "Logistic", "VDP", "I_Pendulum" or "Hénon-Heiles")
num_meth = "Integral_Euler"    # Choice of the numerical method ("Integral_Euler" or "Integral_Euler_order2")
step_h = [0.001, 0.1]          # Interval where time steps are selected for training
step_eps =  [1 , 1]      # Interval where high oscillations parameters are selected for training
T_simul = 1                   # Time for ODE's simulation
h_simul = 0.01                 # Time step used for ODE simulation
eps_simul = 0.1                # High oscillation parameter used for ODE simulation

# AI parameters [adjust]

K_data = 10000           # Quantity of data
R = 2                  # Amplitude of data in space (i.e. space data will be selected in the box [-R,R]^d)
p_train = 0.8          # Proportion of data for training
N_terms = 1            # Number of terms in the perturbation (MLP's)
HL = 2                 # Hidden layers per MLP
zeta = 200             # Neurons per hidden layer
n_F = 10                # Number of Fourier coefficients in the Symbolic regression part
n_eps = 1              # Number of epsilon powers coefficients in the Symbolic regression part
n_d = 5                # Degree of polynomials involved in symbolic regression part
alpha = 1e-3           # Learning rate for gradient descent
Lambda = 1e-9          # Weight decay
BS = 500              # Batch size (for mini-batching)
N_epochs = 1000         # Epochs
N_epochs_print = 100    # Epochs between two prints of the Loss value

print(150 * "-")
print("Using of MLP's of dynamical system - Modified equation - Variable steps of time - Application to highly oscillatory ODE's")
print(150 * "-")

print("   ")
print(150 * "-")
print("Parameters:")
print(150 * "-")
print('    # Maths parameters:')
print('        - Dynamical system studied:', dyn_syst)
print('        - Numerical method:', num_meth)
print("        - Interval where time steps are selected for training:", step_h)
print("        - Interval where high oscillations parameters are selected for training:" , step_eps)
print("        - Time for ODE's simulation:", T_simul)
print("        - Time step used for ODE simulation:", h_simul)
print("        - High oscillation parameter used for ODE simulation:", eps_simul)
print("    # AI parameters:")
print("        - Data's number:", K_data)
print("        - Amplitude of data in space:", R)
print("        - Proportion of data for training:", format(p_train, '.1%'))
print("        - Numbers of terms in the perturbation (MLP's):", N_terms)
print("        - Hidden layers per MLP:", HL)
print("        - Neurons on each hidden layer:", zeta)
print("        - Number of Fourier coefficients in the Symbolic regression part:" , n_F)
print("        - Number of epsilon powers coefficients in the Symbolic regression part:" , n_eps)
print("        - Degree of polynomials involved in symbolic regression part:" , n_d)
print("        - Learning rate:", format(alpha, '.2e'))
print("        - Weight decay:", format(Lambda, '.2e'))
print("        - Batch size (mini-batching for training):", BS)
print("        - Epochs:", N_epochs)
print("        - Epochs between two prints of the Loss value:", N_epochs_print)


# Dimension of the problem

if dyn_syst == "Logistic":
    d = 1
if dyn_syst == "Pendulum" or dyn_syst == "VDP" or dyn_syst == "I_Pendulum":
    d = 2
if dyn_syst == "Hénon-Heiles":
    d = 4

# Initial data for study of trajectories

def y0_start(dyn_syst):
    """Gives the initial data (vector) for ODE's integration and study of trajectories
    Imputs:
    - dyn_syst: character string - Dynamical system studied"""
    if d == 1:
        if dyn_syst == "Logistic":
            Y0_start = np.array([1.5])
    if d == 2:
        if dyn_syst == "Pendulum" or dyn_syst == "VDP" or dyn_syst == "I_Pendulum":
            Y0_start = np.array([0.5, 1.5])
    if d == 4:
        if dyn_syst == "Hénon-Heiles":
            Y0_start = np.array([0,0,0.5,0.5])
    return Y0_start


# Number of iterations for approximations of Phi

if num_meth == "Forward Euler":
    n_iter = 1
if num_meth == "MidPoint" or num_meth == "RK2":
    n_iter = 2


class DynamicSyst:
    """Expressions of vector fields"""
    def f(t, y , eps):
        """Function involved in the averaging process:
         Inputs:
         - t: Float - Time
         - y: Array of shape (1,) - Space variable
         - eps: Float - High oscillation parameter"""

        y = np.array(y).reshape(d, )
        z = np.zeros_like(y)
        if dyn_syst == "Logistic":
            z = (y * (1 - y) + np.array([np.sin(t/eps)]))
        if dyn_syst == "Pendulum":
            z[0] = -np.sin(y[1])
            z[1] = y[0]
        if dyn_syst == "VDP":
            y1 , y2 , tau = y[0] , y[1] , t/eps
            #z1 = (1 / 8) * y2 * (4 - y1 ** 2 - y2 ** 2) - (1 / 4) * y1 * (1 - y2 ** 2) * np.cos(2 * tau) - (1 / 8) * y2 * (-y1 ** 2 + y2 ** 2 - 2) * np.sin(2 * tau) + (1 / 16) * y1 * (y1 ** 2 - 3 * y2 ** 2) * np.cos(4 * tau) + (1 / 16) * y2 * (3 * y1 ** 2 - y2 ** 2) * np.sin(4 * tau)
            #z1 = (1/4)*(y2 + y2*np.cos(2*tau) - y1*np.sin(2*tau))*(2-y1**2-y2**2-(y1**2-y2**2)*np.cos(2*tau) - 2*y1*y2*np.sin(2*tau))
            #z[0] = (1/4)*( y1*(2-y1**2-y2**2)   -   y1*(2-2*y2**2)*np.cos(2*tau)   -   y2*(2+y1**2-y2**2)*np.sin(2*tau)   +   y1*(y1**2-y2**2)*np.cos(2*tau)**2   +   y2*(y1**2-y2**2)*np.cos(2*tau)*np.sin(2*tau)      +   2*y1**2*y2*np.cos(2*tau)*np.sin(2*tau)   +   2*y1*y2**2*np.sin(2*tau)**2)
            #z[0] = (1/4)*( y1*(2-(y1**2+y2**2)/2)   -   2*y1*(1-y2**2)*np.cos(2*tau)   -   y2*(2+y1**2-y2**2)*np.sin(2*tau)   +  (1/2)*y1*(y1**2-y2**2)*np.cos(4*tau)   +   (1/2)*y2*(3*y1**2-y2**2)*np.sin(4*tau)    - y1*y2**2*np.cos(4*tau))
            #z[0] = (1/8)*y1*(4-(y1**2+y2**2))   -   (1/2)*y1*(1-y2**2)*np.cos(2*tau)   -   (1/4)*y2*(2+y1**2-y2**2)*np.sin(2*tau)   +  (1/8)*y1*(y1**2-3*y2**2)*np.cos(4*tau)   +   (1/8)*y2*(3*y1**2-y2**2)*np.sin(4*tau)
            #z[0] = -np.sin(t/eps)*(1 - (y[0]*np.cos(t/eps)+y[1]*np.sin(t/eps))**2)*(-y[0]*np.sin(t/eps)+y[1]*np.cos(t/eps))
            #z1 = (1 / 8) * y2 * (4 - (y1 ** 2 + y2 ** 2)) + (1 / 2) * y2 * (1 - y1 ** 2) * np.cos(2 * tau) - (
            #            1 / 4) * y1 * (          2 + y2 ** 2 - y1 ** 2) * np.sin(2 * tau) - (1 / 8) * y2 * (3*y1 ** 2 - y2 ** 2) * np.cos(4 * tau) + (1 / 8) * y1 * (y1 ** 2 - 3*y2 ** 2) * np.sin(4 * tau)
            z[0] = -np.sin(t/eps)*(1 - (y[0]*np.cos(t/eps)+y[1]*np.sin(t/eps))**2)*(-y[0]*np.sin(t/eps)+y[1]*np.cos(t/eps))
            z[1] = np.cos(t/eps)*(1 - (y[0]*np.cos(t/eps)+y[1]*np.sin(t/eps))**2)*(-y[0]*np.sin(t/eps)+y[1]*np.cos(t/eps))
            #print(z[1] - z1)
        if dyn_syst == "I_Pendulum":
            y1, y2, tau = y[0], y[1], t / eps
            z[0] = y2 + np.sin(t/eps)*np.sin(y1)
            z[1] = np.sin(y1) - (1/2)*np.sin(t/eps)**2*np.sin(2*y1) - np.sin(t/eps)*np.cos(y1)*y2
        if dyn_syst == "Hénon-Heiles":
            z[0] = 2 * np.sin(t/eps) * (y[0] * np.cos(t/eps) + y[2] * np.sin(t/eps)) * y[1]
            z[1] = y[3]
            z[2] = -2 * np.cos(t/eps) * (y[0] * np.cos(t/eps) + y[2] * np.sin(t/eps)) * y[1]
            z[3] = -1 * (y[0] * np.cos(t/eps) + y[2] * np.sin(t/eps)) ** 2 + (3 / 2) * y[1] ** 2 - y[1]
        return z

    def f_avg(y):
        """Average function involved in the averaging process:
         Inputs:
         - y: Array of shape (1,) - Space variable"""

        y = np.array(y).reshape(d, )
        z = np.zeros_like(y)
        if dyn_syst == "Logistic":
            z = y * (1 - y)
        if dyn_syst == "Pendulum":
            z[0] = -np.sin(y[1])
            z[1] = y[0]
        if dyn_syst == "VDP":
            y1 , y2 = y[0] , y[1]
            z[0] = (1/8)*(4-y1**2-y2**2)*y1
            z[1] = (1/8)*(4-y1**2-y2**2)*y2
        if dyn_syst == "I_Pendulum":
            y1, y2 = y[0], y[1]
            z[0] = y2
            z[1] = np.sin(y1) - (1/4)*np.sin(2*y1)
        if dyn_syst == "Hénon-Heiles":
            z[0] = 2 * np.sin(t/eps) * (y[0] * np.cos(t/eps) + y[2] * np.sin(t/eps)) * y[1]
            z[1] = y[3]
            z[2] = -2 * np.cos(t/eps) * (y[0] * np.cos(t/eps) + y[2] * np.sin(t/eps)) * y[1]
            z[3] = -1 * (y[0] * np.cos(t/eps) + y[2] * np.sin(t/eps)) ** 2 + (3 / 2) * y[1] ** 2 - y[1]
        return z

    def g_1(t , s , y , eps):
        """Function used in the Integral Euler scheme of order 2 in the second term.
        Inputs:
        - t: Float: First time variable
        - s: Float: Second time variable
        - y: Array of shape (d,) - Space variable
        - eps: Float - High oscillation parameter
        """
        eta = 1e-5 # Small parameter for finite difference approximation
        g1 = (DynamicSyst.f(t , y + eta*DynamicSyst.f(s , y , eps) , eps) - DynamicSyst.f(t , y - eta*DynamicSyst.f(s , y , eps) , eps))/(2*eta)
        return g1

    def f_NN(t, y, eps):
        """Function involved in the averaging process, adapted to tensors:
         Inputs:
         - t: Float - Time
         - y: Array of shape (1,) - Space variable
         - eps: Float - High oscillation parameter"""

        z = torch.zeros_like(y)
        def sin(tau):
            return torch.sin(tau)
        def cos(tau):
            return torch.cos(tau)

        if dyn_syst == "Logistic":
            z = (y * (1 - y) + sin(t/eps))
        if dyn_syst == "Pendulum":
            z[0,:] = -sin(y[1,:])
            z[1,:] = y[0,:]
        if dyn_syst == "VDP":
            y1, y2 = y[0, :], y[1, :]
            tau = t/eps
            #z1 = (1/8)*y1*(4-y1**2-y2**2) - (1/4)*y1*(1-y2**2)*cos(2*tau) - (1/8)*y2*(-y1**2 + y2**2 - 2)*sin(2*tau) + (1/16)*y1*(y1**2-3*y2**2)*cos(4*tau) + (1/16)*y2*(3*y1**2-y2**2)*sin(4*tau)
            #z2 = -sin(t/eps)*(1 - (y1*cos(t/eps)+y2*sin(t/eps))**2)*(-y1*sin(t/eps)+y2*cos(t/eps))
            #print(z1-z1)
            #z[0,:] = (1 / 8) * y1 * (4 - (y1 ** 2 + y2 ** 2)) - (1 / 2) * y1 * (1 - y2 ** 2) * np.cos(2 * tau) - (1 / 4) * y2 * (
            #            2 + y1 ** 2 - y2 ** 2) * np.sin(2 * tau) + (1 / 8) * y1 * (y1 ** 2 - 3 * y2 ** 2) * np.cos(
            #    4 * tau) + (1 / 8) * y2 * (3 * y1 ** 2 - y2 ** 2) * np.sin(4 * tau)`
            #z[1, :] = (1 / 8) * y2 * (4 - (y1 ** 2 + y2 ** 2)) + (1 / 2) * y2 * (1 - y1 ** 2) * np.cos(2 * tau) - (
            #            1 / 4) * y1 * (          2 + y2 ** 2 - y1 ** 2) * np.sin(2 * tau) - (1 / 8) * y2 * (3*y1 ** 2 - y2 ** 2) * np.cos(4 * tau) + (1 / 8) * y1 * (y1 ** 2 - 3*y2 ** 2) * np.sin(4 * tau)
            z[0,:] = -sin(t/eps)*(1 - (y1*cos(t/eps)+y2*sin(t/eps))**2)*(-y1*sin(t/eps)+y2*cos(t/eps))
            z[1,:] = cos(t/eps)*(1 - (y1*cos(t/eps)+y2*sin(t/eps))**2)*(-y1*sin(t/eps)+y2*cos(t/eps))
        if dyn_syst == "I_Pendulum":
            y1, y2 = y[0, :], y[1, :]
            z[0] = y2 + sin(t / eps) * sin(y1)
            z[1] = sin(y1) - (1 / 2) * sin(t / eps) ** 2 * sin(2 * y1) - sin(t / eps) * cos(y1) * y2
        if dyn_syst == "Hénon-Heiles":
            y1, y2, y3, y4 = y[0, :], y[1, :], y[2, :], y[3, :]
            z[0,:] = 2 * sin(t/eps) * (y1 * cos(t/eps) + y3 * sin(t/eps)) * y2
            z[1,:] = y4
            z[2,:] = -2 * cos(t/eps) * (y1 * cos(t/eps) + y3 * sin(t/eps)) * y2
            z[3,:] = -1 * (y1 * cos(t/eps) + y3 * sin(t/eps)) ** 2 + (3 / 2) * y2 ** 2 - y2
        return z

    def f_avg_NN(y):
        """Average vector field, for tensor inputs.
        Inputs:
        - y: Tensor of shape (d,n)"""
        z = torch.zeros_like(y)
        if dyn_syst == "Logistic":
            z = y * (1 - y)
        if dyn_syst == "VDP":
            y1, y2 = y[0, :], y[1, :]
            z[0,:] = -(1/8)*(y1**2+y2**2-4)*y1
            z[1,:] = -(1/8)*(y1**2+y2**2-4)*y2
        if dyn_syst == "I_Pendulum":
            y1, y2 = y[0, :], y[1, :]
            z[0, :] = y2
            z[1, :] = torch.sin(y1) - (1/4)*torch.sin(2*y1)
        if dyn_syst == "Hénon-Heiles":
            y1, y2, y3, y4 = y[0, :], y[1, :], y[2, :], y[3, :]
            z[0, :] = y3 * y2
            z[1, :] = y4
            z[2, :] = - y1 * y2
            z[3, :] = -y1**2 - y3**2 + (3/2)*y2**2 - y2
        return z


class DynamicSystODE(DynamicSyst):
    """Integration of ODE's"""

    def solveODE(y0 , Ti = 0 ,Tf=T_simul, h=h_simul, eps=eps_simul , rel_tol=1e-9, abs_tol=1e-9):
        """Solving of the ODE y'(t)=f(t/eps,y(t)) over an time interval [0,T] with step time h by using DOP853 method
                Inputs:
                - y0: Array of shape (d,) - Initial data for ODE solving
                - Ti: Float - Initial time (default: 0)
                - Tf: Float - Final time (default: T_simul)
                - h: Float - Time step (default: h_simul)
                - eps: Float - High oscillation parameter (default: eps_simul)
                - rel_tol: Float - Relative error tolerance (default: 1e-9)
                - abs_tol: Float - Absolute error tolerance (default: 1e-9)
                => Returns an array of shape (d,n) where n is the number of steps for numerical integration"""
        def dynamics(t,y):
            """vector field for ODE integration.
            Inputs:
             - t: Float - Time variable
             - y: Array of shape (d,) - Space variable"""
            return DynamicSyst.f(t,y,eps)

        return solve_ivp(dynamics, (Ti, Tf+h), y0, method='DOP853', t_eval=np.arange(Ti, Tf, h), rtol=rel_tol, atol=abs_tol).y

    def solveODE_Num(y0 , F , Ti = 0 , Tf=T_simul , h=h_simul , meth = num_meth):
        """Numerical resolution of the nonstiff ODE y'(t) = F(t , y(t)).
        Inputs:
        - y0: Array of shape (d,) - Initial condition
        - F: Vector field used for numerical integration
        - n_app: Int - Iteration for approximation of averaged field and mapping Phi.
        - Ti: Float - Initial time (default: 0)
        - Tf: Float - Final time (default: T_simul)
        - h: Float - Time step for ODE simulation (default: h_simul)
        - eps: Float - High oscillation parameter (default: eps_simul)
        - meth: Str - Name of the numerical method. (default: num_meth)"""
        Y = y0.reshape(d, 1)
        y = y0
        TT = np.arange(Ti, Tf, h)
        for t in TT[1:]:
            if meth == "Forward Euler":
                y = y + h * F(t-h, y)
                Y = np.concatenate((Y, y.reshape(d, 1)), axis=1)
            if meth == "Integral_Euler":
                y = y + F(t-h, y)
                Y = np.concatenate((Y, y.reshape(d, 1)), axis=1)
            if meth == "Integral_Euler_order2":
                y = y + F(t-h, y)
                Y = np.concatenate((Y, y.reshape(d, 1)), axis=1)
            if meth == "MidPoint":
                def func_iter(x):
                    x = x.reshape(d, )
                    z = y + h * F(t - h/2, (x + y) / 2)
                    z = z.reshape(d,)
                    return z
                Niter = 5
                x = y
                for k in range(Niter):
                    x = func_iter(x)
                y = x
                Y = np.concatenate((Y, y.reshape(d, 1)), axis=1)
            if meth == "RK2":
                y = y + h * F(t - h, y + (h / 2) * F(t - h / 2, y))
                Y = np.concatenate((Y, y.reshape(d, 1)), axis=1)

        return Y

    def solveODE_Data(K=K_data, p=p_train, h_data=step_h, eps_data=step_eps):
        """production of a set of initial data y0 and final data y1 = flow-h(y0) associated to the dynamical system y'(t)=f_PB(t/eps,y(t))
        associated to the Pullback system
        Inputs:
        - n_appr: Integer - Number of iterations required for approximation of Phi and its derivatives
        - K: Data's number (default: K_data)
        - p: Proportion of data for training (default: p_train)
        - h_data: List - Interval where time steps are chosen for training (default: step_h)
        - eps_data: List - Interval where steps of time are chosen for training (default: step_h)
        Denote K0 := int(p*K) the number of data for training
        => Returns the tuple (Y0_train,Y0_test,Y1_train,Y1_test,h_train,h_test) where:
            - Y0_train is a tensor of shape (d,K0) associated to the initial data for training
            - Y0_test is a tensor of shape (d,K-K0) associated to the initial data for test
            - Y1_train is a tensor of shape (d,K0) associated to the final data for training
            - Y1_test is a tensor of shape (d,K-K0) associated to the final data for test
            - h_train is a tensor of shape (1,K0) associated to data of steps of time for training
            - h_train is a tensor of shape (1,K-K0) associated to data of steps of time for test
            Each column of the tensor Y1_* correspopnds to the flow at h_* of the same column of Y0_*
            Initial data are uniformly chosen in [-R,R]^d (excepted for Rigid Body, in a spherical crown)"""

        start_time_data = time.time()

        print(" ")
        print(150 * "-")
        print("Data creation...")
        print(150 * "-")

        K0 = int(p * K)
        YY0 = np.random.uniform(low=-R, high=R, size=(d, K))
        tt0 = np.random.uniform(low=0 , high=2*np.pi , size = (1,K))
        #runtt0 = np.zeros_like(tt0)
        if dyn_syst == "Logistic":
            YY0 = np.abs(YY0)
        #YY1 , YY2 = np.zeros((d, K)) , np.zeros((d, K))
        YY1 = np.zeros((d, K))
        hh = np.exp(np.random.uniform(low=np.log(h_data[0]), high=np.log(h_data[1]), size=(1, K)))
        #eeps = np.exp(np.random.uniform(low=np.log(eps_data[0]), high=np.log(eps_data[1]), size=(1, K)))
        eeps = np.exp(np.random.uniform(low=np.log(eps_simul), high=np.log(eps_simul), size=(1, K)))

        pow = max([int(np.log10(K) - 1), 3])
        pow = min([pow, 6])
        for k in range(K):
            end_time_data = start_time_data + (K / (k + 1)) * (time.time() - start_time_data)
            end_time_data = datetime.datetime.fromtimestamp(int(end_time_data)).strftime(' %Y-%m-%d %H:%M:%S')
            print(" Loading :  {} % \r".format(str(int(10 ** (pow) * (k + 1) / K) / 10 ** (pow - 2)).rjust(3)), " Estimated time for ending : " + end_time_data, " - ", end="")
            #YYY = DynamicSystODE.solveODE_Pull_back(y0=YY0[:, k], n_app=n_appr, T=3 * hh[0, k], h=hh[0, k] , eps = eeps[0,k])#[:, 1]
            #YYY = DynamicSystODE.solveODE_Pull_back(y0=YY0[:, k], n_app=n_appr, T=3 * hh[0, k] * eeps[0, k], h=hh[0, k] * eeps[0, k] , eps = eeps[0,k])#[:, 1]
            YYY = DynamicSystODE.solveODE(y0=YY0[:, k] , Ti=tt0[0,k] ,  Tf = tt0[0,k] + 2 * hh[0, k], h=hh[0, k] , eps = eeps[0,k])#[:, 1]
            #YYY = DynamicSystODE.solveODE(y0=YY0[:, k], T=3 * hh[0, k] * eeps[0, k], h=hh[0, k] * eeps[0, k] , eps = eeps[0,k])#[:, 1]
            YY0[:, k] , YY1[:, k] = YYY[:, 0] , YYY[:, 1]

        Y0_train = torch.tensor(YY0[:, 0:K0])
        Y0_test = torch.tensor(YY0[:, K0:K])
        Y1_train = torch.tensor(YY1[:, 0:K0])
        Y1_test = torch.tensor(YY1[:, K0:K])
        tt0_train = torch.tensor(tt0[:, 0:K0])
        tt0_test = torch.tensor(tt0[:, K0:K])
        h_train = torch.tensor(hh[:, 0:K0])
        h_test = torch.tensor(hh[:, K0:K])
        eps_train = torch.tensor(eeps[:, 0:K0])
        eps_test = torch.tensor(eeps[:, K0:K])

        print("Computation time for data creation (h:min:s):",
              str(datetime.timedelta(seconds=int(time.time() - start_time_data))))
        return (Y0_train, Y0_test, Y1_train, Y1_test, tt0_train ,tt0_test , h_train, h_test, eps_train, eps_test)


class NN(nn.Module, DynamicSystODE):
    def __init__(self):
        super().__init__()
        #zeta_bis = int(zeta / N_terms)
        zeta_bis = int(zeta)
        if num_meth == "Forward Euler":
            self.RF = nn.ModuleList([nn.Linear(d + 3, zeta_bis), nn.Tanh() ] + (HL - 1) * [nn.Linear(zeta_bis, zeta_bis), nn.Tanh()] + [nn.Linear(zeta_bis, d, bias=True)])
        if num_meth == "Integral_Euler":
            self.RF = nn.ModuleList([nn.Linear(d + 3, zeta_bis), nn.Tanh() ] + (HL - 1) * [nn.Linear(zeta_bis, zeta_bis), nn.Tanh()] + [nn.Linear(zeta_bis, d, bias=True)])

    def forward(self, tau, x, eps, h):
    #def forward(self, x, h):
        """tau is a tensor or a float (linked with data h and eps used for training of scalar for numerical simulation)
        x is a tensor (space variables)
        h is a tensor or a float (data h used for training of scalar for numerical simulation)
        eps is a tensor or a float (data eps used for training of scalar for numerical simulation)"""
        x = x.T
        tau = torch.tensor(tau).T
        eps = torch.tensor(eps).T
        h = torch.tensor(h).T
        x = x.float()

        ONE = torch.ones_like(x[:, 0]).reshape(x[:, 0].size()[0], 1)

        if num_meth == "Forward Euler":
            x0 = DynamicSyst.f_NN(tau.T, x.T, eps.T).T
            xRF = torch.cat((torch.cos(tau) , torch.sin(tau) , x , h), dim=1)
            #xRF = torch.cat((tau , x , h), dim=1)
            for i, module in enumerate(self.RF):
                #xRF = (xRF - torch.mean(xRF))/torch.std(xRF)
                xRF = module(xRF)
            return (x0 + h*xRF).T
        if num_meth == "Integral_Euler":
            x0_avg = DynamicSyst.f_avg_NN(x.T).T
            x_r = torch.zeros_like(x0_avg)
            xi , omega = npp.polynomial.legendre.leggauss(10)
            tau_0 = tau + torch.round(h/(2*np.pi*eps))*2*np.pi
            tau_1 = tau + h/eps
            xi = [(tau_1-tau_0)/2*ksi + (tau_0+tau_1)/2 for ksi in xi]
            for i in range(10):
                x_r += eps*omega[i]*(tau_1-tau_0)/2*(DynamicSyst.f_NN((eps*xi[i]).T , x.T , eps.T) - DynamicSyst.f_avg_NN(x.T)).T
            #xRF = torch.cat((eps*tau , x , h), dim=1)
            #xRF = torch.cat((tau , x , h), dim=1)
            xRF = torch.cat((torch.cos(tau) , torch.sin(tau) , x , h), dim=1)
            for i, module in enumerate(self.RF):
                #xRF = (xRF - torch.mean(xRF))/torch.std(xRF)
                xRF = module(xRF)

            return (h*x0_avg + x_r + h**2*xRF).T


        if num_meth == "MidPoint" or num_meth == "RK2":

            if N_terms >= 2:
                x1 = torch.cat((torch.cos(tau), torch.sin(tau), x, eps), dim=1)
                for i, module in enumerate(self.f1):
                    x1 = module(x1)
                x0 = x0 + h * x1
            if N_terms >= 3:
                x2 = torch.cat((torch.cos(tau), torch.sin(tau), x, eps), dim=1)
                for i, module in enumerate(self.f2):
                    x2 = module(x2)
                x0 = x0 + h * x1 + h ** 2 * x2
            if N_terms >= 4:
                x3 = torch.cat((torch.cos(tau), torch.sin(tau), x, eps), dim=1)
                for i, module in enumerate(self.f3):
                    x3 = module(x3)
                x0 = x0 + h * x1 + h ** 2 * x2 + h ** 3 * x3

            xRF = torch.cat((torch.cos(tau), torch.sin(tau), x, eps, h), dim=1)
            xRR = torch.cat((tau, x, h), dim=1)

            for i, module in enumerate(self.RF):
                xRF = module(xRF)
            for i, module in enumerate(self.RR):
                xRR = module(xRR)

            return (x0 + (h ** (N_terms+1)) * (xRF + eps ** (n_iter + 1) * xRR)).T


class Train(NN, DynamicSystODE):
    """Training of the neural network, depends on the numerical method chosen
    Choice of the numerical method:
        - Forward Euler
        - MidPoint
        - RK2"""

    def Loss(self, Y0, Y1, tt0 , eps, h, model, meth=num_meth):
        """Computes the Loss function between two series of data Y0 and Y1 according to the numerical method
        Inputs:
        - Y0: Tensor of shape (d,n)
        - Y1: Tensor of shape (d,n)
        - eps: Tensor of shape (1,n)
        - tt0: Tensor of shape (1,n)
        - h: Tensor of shape (1,n)
        - model: Neural network which will be optimized
        - meth: Character string - Numerical method used in order to compute predicted values
        Computes a predicted value Y1hat which is a tensor of shape (d,n) and returns the mean squared error between Y1hat and Y1
        => Returns a tensor of shape (1,1)"""
        Y0 = torch.tensor(Y0, dtype=torch.float32)
        Y0.requires_grad = True
        Ymeth = torch.zeros_like(Y0)
        Ymeth.requires_grad = True
        tt0 = torch.tensor(tt0, dtype=torch.float32)
        tt0.requires_grad = True
        eps = torch.tensor(eps, dtype=torch.float32)
        eps.requires_grad = True
        h = torch.tensor(h, dtype=torch.float32)
        h.requires_grad = True
        if meth == "Forward Euler":
            Ymeth = Y0
            Y1hat = Y0 + h*model(tt0/eps , Y0 , eps , h)
            #Y1hat = Y0 + h*model(Ymeth ,h)
            #Y1hat = Y0 + h*model(Ymeth, eps ,h)
            #id = 90
            #print(" ")
            #print(("Train:",(Y1hat[:,id]-Y1[:,id])**2))
            #print(("Euler",(Y0[:,id]+h[:,id]*DynamicSyst.Pull_back_NN(h[:,id]/eps[:,id] , Y0[:,id] , eps[:,id] , h[:,id])-Y1[:,id])**2))
            #Y1hat = Y0 + eps * h * model(h, Y0, eps ,h*eps)
            #loss = (((eps / h) ** 2) * ((Y1hat - Y1) ** 2)).mean()
            loss = (((eps / h) ** 4) * ((Y1hat - Y1).abs()**2)).mean()
        if meth == "Integral_Euler":
            Ymeth = Y0
            Y1hat = Y0 + model(tt0/eps , Y0 , eps , h)
            loss = (((1 / h) ** 4) * ((Y1hat - Y1).abs()**2)).mean()
        if meth == "MidPoint":
            Ymeth = (Y0 + Y1) / 2
            Y1hat = Y0 + h*model((tt0+h/2)/eps , Ymeth , eps , h)
            loss = ((1 / h ** 6) * (Y1hat - Y1) ** 2).mean()
        if meth == "RK2":
            Y1hat = Y0 + h * model( 3*h/(2*eps) , Y0 + (h / 2) * model( h/eps , Y0 , eps , h) , eps , h)
            loss = ((1 / h ** 6) * (Y1hat - Y1) ** 2).mean()

        return loss

    def train(self, model, Data, K=K_data, p=p_train, Nb_epochs=N_epochs, Nb_epochs_print=N_epochs_print, BSZ=BS):
        """Makes the training on the data
        Inputs:
        - model: Neural network which will be optimized
        - Data: Tuple of tensors - Set of data created
        - K: Integer - Number of data
        - p: Float - Proportion of data used for training (default: p_train)
        - Nb_epochs: Integer - Number of epochs for training (dafault: N_epochs)
        - Nb_epochs_print: Integer - Number of epochs between two prints of the value of the Loss (default: N_epochs_print)
        - BSZ: Integer - size of the batches for mini-batching
        => Returns the lists Loss_train and Loss_test of the values of the Loss respectively for training and test,
        and best_model, which is the best apporoximation of the modified field computed"""

        start_time_train = time.time()

        print(" ")
        print(150 * "-")
        print("Training...")
        print(150 * "-")

        Y0_train = Data[0]
        Y0_test = Data[1]
        Y1_train = Data[2]
        Y1_test = Data[3]
        tt0_train = Data[4]
        tt0_test = Data[5]
        h_train = Data[6]
        h_test = Data[7]
        eps_train = Data[8]
        eps_test = Data[9]
        optimizer = optim.AdamW(model.parameters(), lr=alpha, betas=(0.9, 0.999), eps=1e-8, weight_decay=Lambda, amsgrad=True)  # Algorithm AdamW
        best_model, best_loss_train, best_loss_test = model, np.infty, np.infty  # Best model (best minimizer of the Loss function)
        Loss_train = []  # List for loss_train values
        Loss_test = []  # List for loss_test values

        for params in model.parameters():
            params = 0*params


        for epoch in range(Nb_epochs + 1):
            #b = np.random.randint(low=1,high=2)
            #i1 = np.random.randint(low=0 , high=Y0_train.shape[1]-1 , size=BS)
            for ixs in torch.split(torch.arange(Y0_train.shape[1]), BS):
            #for ixs in [torch.tensor(i1)]:
                optimizer.zero_grad()
                #model.train()
                Y0_batch = Y0_train[:, ixs]
                Y1_batch = Y1_train[:, ixs]
                tt0_batch = tt0_train[:,ixs]
                eps_batch = eps_train[:,ixs]
                h_batch = h_train[:, ixs]
                loss_train = self.Loss(Y0_batch, Y1_batch, tt0_batch , eps_batch, h_batch, model)
                loss_train.backward()
                optimizer.step()  # Optimizer passes to the next epoch for gradient descent

            loss_test = self.Loss(Y0_test, Y1_test, tt0_test , eps_test, h_test, model)

            if loss_train < best_loss_train:
                best_loss_train = loss_train
                best_loss_test = loss_test
                best_model = copy.deepcopy(model)
                # best_model = model

            Loss_train.append(loss_train.item())
            Loss_test.append(loss_test.item())

            if epoch % Nb_epochs_print == 0:  # Print of Loss values (one print each N_epochs_print epochs)
                end_time_train = start_time_train + ((N_epochs + 1) / (epoch + 1)) * (time.time() - start_time_train)
                end_time_train = datetime.datetime.fromtimestamp(int(end_time_train)).strftime(' %Y-%m-%d %H:%M:%S')
                print('    Step', epoch, ': Loss_train =', format(loss_train, '.4E'), ': Loss_test =', format(loss_test, '.4E'), " -  Estimated end:", end_time_train)

        print("Loss_train (final)=", format(best_loss_train, '.4E'))
        print("Loss_test (final)=", format(best_loss_test, '.4E'))

        print("Computation time for training (h:min:s):", str(datetime.timedelta(seconds=int(time.time() - start_time_train))))

        return (Loss_train, Loss_test, best_model)


class Integrate(Train , DynamicSystODE):

    def integrate(self, model, name, save_fig):
        """Prints the values of the Loss along the epochs, trajectories and errors.
        Inputs:
        - model: Best model learned during training, Loss_train and Loss_test
        - name: Character string - Potential name of the graph
        - save_fig: Boolean - Saves or not the figure"""

        def write_size():
            """Changes the size of writings on all windows"""
            axes = plt.gca()
            axes.title.set_size(7)
            axes.xaxis.label.set_size(7)
            axes.yaxis.label.set_size(7)
            plt.xticks(fontsize=7)
            plt.yticks(fontsize=7)
            plt.legend(fontsize=7)
            pass

        def write_size3D():
            """Changes the size of writings on all windows - 3d variant"""
            axes = plt.gca()
            axes.title.set_size(7)
            axes.xaxis.label.set_size(7)
            axes.yaxis.label.set_size(7)
            axes.zaxis.label.set_size(7)
            plt.xticks(fontsize=7)
            plt.yticks(fontsize=7)
            axes.zaxis.set_tick_params(labelsize=7)
            plt.legend(fontsize=7)
            pass

        start_time_integrate = time.time()

        Model, Loss_train, Loss_test = model[0], model[1], model[2]

        print(" ")
        print(150 * "-")
        print("Integration...")
        print(150 * "-")

        fig = plt.figure()

        ax = fig.add_subplot(2, 1, 2)
        plt.plot(range(len(Loss_test)), Loss_train, color='green', label='$Loss_{train}$')
        plt.plot(range(len(Loss_train)), Loss_test, color='red', label='$Loss_{test}$')
        plt.grid()
        plt.legend()
        plt.yscale('log')
        plt.title('Evolution of the Loss function (MLP)')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        write_size()

        def Fhat(t, y):
            """Vector fied learned with the neural network, adapted for ODE integration
            Inputs:
            - t: Float - Time
            - y: Array of shape (d,) - Space variable"""
            y = torch.tensor(y).reshape(d, 1)
            y.requires_grad = True
            t_tensor = torch.tensor([[t]]).float()
            h_tensor = torch.tensor([[h_simul]]).float()
            eps_tensor = torch.tensor([[eps_simul]]).float()
            z = Model(t_tensor/eps_tensor , y , eps_tensor , h_tensor)
            #z = Model(y , h_tensor)
            z = z.detach().numpy()
            z = np.array(z, dtype=np.float64)
            return z.reshape(d, )

        def Fsimul(t ,y):
            """Vector fied of the ODE, adapted for ODE integration
            Inputs:
            - t: Float - Time
            - y: Array of shape (d,) - Space variable"""
            if num_meth == "Forward Euler":
                return DynamicSyst.f(t , y , eps_simul)
            if num_meth == "Integral_Euler":
                yy = 0*y
                xi , omega = npp.polynomial.legendre.leggauss(15)
                xi = [(h_simul/2)*ksi + (t+h_simul/2) for ksi in xi]
                for i in range(15):
                    yy += (h_simul/2)*omega[i]*DynamicSyst.f(xi[i] , y , eps_simul)
                return yy

        TT = np.arange(0, T_simul, h_simul)

        # Integration with DOP853 (good approximation of exact flow)
        start_time_RK45 = time.time()
        #Y_exact = DynamicSystODE.solveODE(y0=y0_start(dyn_syst), T=T_simul, h=h_simul, eps = eps_simul)
        #Y_exact = DynamicSystODE.solveODE_Pull_back_Exact(y0=y0_start(dyn_syst), n_app=n_iter, T=T_simul, h=h_simul, eps = eps_simul)
        Y_exact = DynamicSystODE.solveODE(y0 = y0_start(dyn_syst) , Ti =0 , Tf = T_simul , h = h_simul , eps = eps_simul)
        print("Integration time of ODE with DOP853 (one trajectory - h:min:s):", datetime.timedelta(seconds=time.time() - start_time_RK45))

        # Integration with numerical method
        start_time_exact = time.time()
        Y_exact_meth = DynamicSystODE.solveODE_Num(y0=y0_start(dyn_syst), F = Fsimul , Ti =0 , Tf = T_simul , h=h_simul, meth=num_meth)
        print("Integration time of ODE with exact field - UA - " + num_meth + " (one trajectory - h:min:s):", str(datetime.timedelta(seconds=time.time() - start_time_exact)))

        # Integration with learned vector field
        start_time_app = time.time()
        Y_app_meth = DynamicSystODE.solveODE_Num(y0=y0_start(dyn_syst), F = Fhat , Ti =0 , Tf = T_simul , h=h_simul, meth=num_meth)
        print("Integration time of ODE with learned field - " + num_meth + " (one trajectory - h:min:s):",str(datetime.timedelta(seconds=time.time() - start_time_app)))

        print("   ")
        # Error computation between trajectory ploted with f for RK45 and f for numerical method
        err_f = np.array([np.linalg.norm((Y_exact - Y_exact_meth)[:, i]) for i in  range((Y_exact - Y_exact_meth).shape[1])])
        Err_f = np.linalg.norm(err_f, np.infty)
        print("Error between trajectories ploted with f for RK45 with f for", num_meth, ":", format(Err_f, '.4E'))

        # Error computation between trajectory ploted with f for RK45 and f_app for numerical method
        err_meth = np.array([np.linalg.norm((Y_exact - Y_app_meth)[:, i]) for i in range((Y_exact - Y_app_meth).shape[1])])
        Err_meth = np.linalg.norm(err_meth, np.infty)
        print("Error between trajectories ploted with f for RK45 with f_app for", num_meth, ":", format(Err_meth, '.4E'))

        f = plt.gcf()
        dpi = f.get_dpi()
        h, w = f.get_size_inches()
        f.set_size_inches(h * 1.5, w * 1.5)
        if d == 1:
            plt.subplot(2, 2, 1)
            plt.title("Trajectories")
            plt.axis('equal')
            plt.plot(TT, Y_exact.squeeze(), color='black', linestyle='dashed',label="$\phi_{nh}^f(y_0)$")
            plt.plot(TT, Y_app_meth.squeeze(), color='green', label="$(\Phi_{h}^{f_{\\theta}})^n(y_0)$")
            plt.plot(TT, Y_exact_meth.squeeze(), color='red', label="$(\Phi_{h}^{f})^n(y_0)$")
            plt.xlabel("$t$")
            plt.ylabel("$y$")
            plt.legend()
            plt.grid()
            write_size()
            plt.subplot(2, 2, 2)
            plt.title("Comparison of local errors")
            plt.yscale('log')
            plt.plot(TT, err_f, color="blue", label="$| (\Phi_{h}^{f})^n(y_0) - \phi_{nh}^f(y_0) |$")
            plt.plot(TT, err_meth, color="orange", label="$| (\Phi_{h}^{f_{app}})^n(y_0) - \phi_{nh}^f(y_0) |$")
            plt.xlabel("t")
            plt.ylabel("Local error")
            plt.legend()
            plt.grid()
            write_size()

            if save_fig == True:
                plt.savefig(name + dtime.now().strftime(' - %Y-%m-%d %H:%M:%S') + '.pdf', dpi=(200))
            else:
                plt.show()

        if d == 2:
            plt.subplot(2, 2, 1)
            plt.title("Trajectories")
            plt.axis('equal')
            plt.plot(Y_exact[0, :], Y_exact[1, :], color='black', linestyle='dashed', label="$\phi_{nh}^f(y_0)$")
            plt.plot(Y_app_meth[0, :], Y_app_meth[1, :], color='green', label="$(\Phi_{h}^{f_{\\theta}})^n(y_0)$")
            plt.plot(Y_exact_meth[0, :], Y_exact_meth[1, :], color='red', label="$(\Phi_{h}^{f})^n(y_0)$")
            plt.xlabel("$y_1$")
            plt.ylabel("$y_2$")
            plt.legend()
            plt.grid()
            write_size()
            plt.subplot(2, 2, 2)
            plt.title("Comparison of local errors")
            plt.yscale('log')
            plt.plot(TT, err_f, color="blue", label="$| (\Phi_{h}^{f})^n(y_0) - \phi_{nh}^f(y_0) |$")
            plt.plot(TT, err_meth, color="orange", label="$| (\Phi_{h}^{f_{app}})^n(y_0) - \phi_{nh}^f(y_0) |$")
            plt.xlabel("t")
            plt.ylabel("Local error")
            plt.legend()
            plt.grid()
            write_size()

            if save_fig == True:
                plt.savefig(name + dtime.now().strftime(' - %Y-%m-%d %H:%M:%S') + '.pdf', dpi=(200))
            else:
                plt.show()

            if dyn_syst == "VDP":
                COS , SIN = np.cos(TT / eps_simul), np.sin(TT / eps_simul)

                plt.figure()
                plt.title("Trajectories after variable change")
                plt.axis('equal')
                plt.plot(COS * Y_app_meth[0, :] + SIN * Y_app_meth[1, :], -SIN * Y_app_meth[0, :] + COS * Y_app_meth[1, :], color='green', label="ML+"+num_meth)
                plt.plot(COS * Y_exact[0, :] + SIN * Y_exact[1, :], -SIN * Y_exact[0, :] + COS * Y_exact[1, :], color="red", linestyle="dashed",label=num_meth)
                plt.xlabel("$q$")
                plt.ylabel("$p$")
                plt.legend()
                plt.grid()
                write_size()
                #plt.show()

                if save_fig == True:
                    plt.savefig(name + "_Variable_change" + dtime.now().strftime(' - %Y-%m-%d %H:%M:%S') + '.pdf', dpi=(200))
                else:
                    plt.show()

            # if save_list == True:
            #     torch.save(TT,"Integrate_Time_RKmodif_Comparison_K_"+str(K))
            #     torch.save(err_f,"Integrate_Error_f_RKmodif_Comparison_K_"+str(K_data))
            #     torch.save(err_meth,"Integrate_Error_f_app_RKmodif_Comparison_K_"+str(K))

        if d == 3:
            ax = fig.add_subplot(2, 2, 1, projection='3d')
            ax.plot(Y_exact[0, :], Y_exact[1, :], Y_exact[2, :], color='black', linestyle="dashed",
                    label="$\phi_{nh}^f(y_0)$")
            ax.plot(Y_app_meth[0, :], Y_app_meth[1, :], Y_app_meth[2, :], color='green', linewidth=1,
                    label="$(\Phi_{h}^{f_{app}})^n(y_0)$")
            ax.plot(Y_exact_meth[0, :], Y_exact_meth[1, :], Y_exact_meth[2, :], color='red', linewidth=1,
                    label="$(\Phi_{h}^{f})^n(y_0)$")
            ax.legend()
            ax.set_xlabel('$y_1$')
            ax.set_ylabel('$y_2$')
            ax.set_zlabel('$y_3$')
            plt.title("Trajectories")
            write_size3D()

            ax = fig.add_subplot(2, 2, 2)
            plt.title("Comparison of local errors")
            plt.plot(TT, err_f, color="blue", label="$| (\Phi_{h}^{f})^n(y_0) - \phi_{nh}^f(y_0) |$")
            plt.plot(TT, err_meth, color="orange", label="$| (\Phi_{h}^{f_{\\theta}})^n(y_0) - \phi_{nh}^f(y_0) |$")
            plt.ylabel("local error")
            plt.yscale('log')
            ax.legend()
            plt.grid()
            write_size()

            if save_list == True:
                torch.save(TT,"Integrate_Time_RKmodif_Comparison_K_"+str(K))
                torch.save(err_f,"Integrate_Error_f_RKmodif_Comparison_K_"+str(K_data))
                torch.save(err_meth,"Integrate_Error_f_app_RKmodif_Comparison_K_"+str(K))

        if d == 4:
            plt.subplot(2, 2, 1)
            plt.title("Trajectories")
            plt.axis('equal')
            plt.plot(Y_exact[1, :], Y_exact[3, :], color='black', linestyle='dashed', label="$\phi_{nh}^f(y_0)$")
            plt.plot(Y_app_meth[1, :], Y_app_meth[3, :], color='green', label="$(\Phi_{h}^{f_{\\theta}})^n(y_0)$")
            plt.plot(Y_exact_meth[1, :], Y_exact_meth[3, :], color='red', label="$(\Phi_{h}^{f})^n(y_0)$")
            plt.xlabel("$y_3$")
            plt.ylabel("$y_1$")
            plt.legend()
            plt.grid()
            write_size()
            plt.subplot(2, 2, 2)
            plt.title("Comparison of local errors")
            plt.yscale('log')
            plt.plot(TT, err_f, color="blue", label="$| (\Phi_{h}^{f})^n(y_0) - \phi_{nh}^f(y_0) |$")
            plt.plot(TT, err_meth, color="orange", label="$| (\Phi_{h}^{f_{app}})^n(y_0) - \phi_{nh}^f(y_0) |$")
            plt.xlabel("t")
            plt.ylabel("Local error")
            plt.legend()
            plt.grid()
            write_size()

            if save_fig == True:
                plt.savefig(name + dtime.now().strftime(' - %Y-%m-%d %H:%M:%S') + '.pdf', dpi=(200))
            else:
                plt.show()

        if dyn_syst == "Hénon-Heiles":
            Y_exact_VC, Y_app_meth_VC , Y_exact_meth_VC = np.zeros_like(Y_exact), np.zeros_like(Y_app_meth) ,  np.zeros_like(Y_exact_meth)
            for n in range(np.size(TT)):
                # VC = np.array([[np.cos(TT[0,n]/eps) , 0 , np.sin(TT[0,n]/eps) , 0] , [0 , 1 , 0 , 0] , [-np.sin(TT[0,n]/eps) , 0 , np.cos(TT[0,n]/eps) , 0] , [0 , 0 , 0 , 1]])
                VC = np.array([[np.cos(TT[n] / eps_simul), 0, np.sin(TT[n] / eps_simul), 0], [0, 1, 0, 0],[-np.sin(TT[n] / eps_simul), 0, np.cos(TT[n] / eps_simul), 0], [0, 0, 0, 1]])
                Y_exact_VC[:, n], Y_app_meth_VC[:, n] , Y_exact_meth_VC[:, n] = VC @ Y_exact[:, n], VC @ Y_app_meth[:, n], VC @ Y_exact_meth[:, n]

            #plt.figure()
            plt.plot(np.squeeze(Y_exact_VC[1, :]), np.squeeze(Y_exact_VC[3, :]), label="Exact solution",color="black",linestyle="dashed")
            plt.plot(np.squeeze(Y_exact_meth_VC[1, :]), np.squeeze(Y_exact_meth_VC[3, :]), label="Num solution", color="red")
            plt.plot(np.squeeze(Y_app_meth_VC[1, :]), np.squeeze(Y_app_meth_VC[3, :]), label="Num solution - app", color="green")
            plt.grid()
            plt.legend()
            plt.xlabel("$q_2$")
            plt.ylabel("$p_2$")
            plt.title("$\epsilon = $" + str(eps_simul))
            plt.axis("equal")

            if save_fig == True:
                plt.savefig(name + "_Variable_change" + dtime.now().strftime(' - %Y-%m-%d %H:%M:%S') + '.pdf',  dpi=(200))
            else:
                plt.show()

        print("Computation time for integration (h:min:s):",str(datetime.timedelta(seconds=int(time.time() - start_time_integrate))))

        pass


class Trajectories(Integrate):
    def traj(self, model, name, save_fig):
        """Prints the global errors according to the step of the numerical method
        Inputs:
        - model: Best model learned during training
        - name: Character string - Potential name of the graph
        - save_fig: Boolean - Saves or not the figure"""

        Model = model[0]

        HH = np.exp(np.linspace(np.log(step_h[0]), np.log(step_h[1]), 11))
        #EEPS = np.exp(np.linspace(np.log(step_eps[0]), np.log(step_eps[1]), 1))
        EEPS = np.exp(np.linspace(np.log(eps_simul), np.log(eps_simul), 1))
        ERR_f , ERR_meth = np.zeros((np.size(EEPS),np.size(HH))) , np.zeros((np.size(EEPS),np.size(HH))) # Global errors

        for i in range(np.size(EEPS)):
            eeps = EEPS[i]
            for j in range(np.size(HH)):
                hh = HH[j]
                print(" - h = {} \r".format(format(hh, '.4E')),"eps = "+format(format(eeps, '.4E')), end="")

                def Fhat(t, y):
                    """Vector fied learned with the neural network, adapted for ODE integration
                    Inputs:
                    - t: Float - Time
                    - y: Array of shape (d,) - Space variable"""
                    y = torch.tensor(y).reshape(d, 1)
                    y.requires_grad = True
                    t_tensor = torch.tensor([[t]]).float()
                    h_tensor = torch.tensor([[hh]]).float()
                    eps_tensor = torch.tensor([[eeps]]).float()
                    z = Model(t_tensor/eps_tensor, y, eps_tensor, h_tensor)
                    # z = Model(y , h_tensor)
                    z = z.detach().numpy()
                    z = np.array(z, dtype=np.float64)
                    return z.reshape(d, )

                def Fsimul(t, y):
                    """Vector fied of the ODE, adapted for ODE integration
                    Inputs:
                    - t: Float - Time
                    - y: Array of shape (d,) - Space variable"""
                    if num_meth == "Forward Euler":
                        return DynamicSyst.f(t, y, eps_simul)
                    if num_meth == "Integral_Euler":
                        yy = hh * DynamicSyst.f_avg(y)
                        xi, omega = npp.polynomial.legendre.leggauss(10)
                        tau_0 = t / eps_simul + 2 * np.pi * np.round(hh / (2 * np.pi * eps_simul))
                        tau_1 = (t + hh) / eps_simul
                        xi = [((tau_1 - tau_0) / 2) * ksi + (tau_1 + tau_0) / 2 for ksi in xi]
                        for i in range(10):
                            yy += eps_simul * ((tau_1 - tau_0) / 2) * omega[i] * (DynamicSyst.f(eps_simul * xi[i], y, eps_simul) - DynamicSyst.f_avg(y))
                        return yy

                # Integration with DOP853 (approximation of the exact flow)
                #Y_exact = DynamicSystODE.solveODE(y0 = y0_start(dyn_syst) , T = T_simul , h = hh , eps = eeps , rel_tol = 1e-15, abs_tol = 1e-15)
                Y_exact = DynamicSystODE.solveODE(y0 = y0_start(dyn_syst) , Ti = 0 , Tf = T_simul , h = hh , eps = eeps)
                norm_sol = np.linalg.norm(np.array([np.linalg.norm((Y_exact)[:, i]) for i in range((Y_exact).shape[1])]),np.infty)  # Norm of the exact solution

                # Integration with UA method
                Y_exact_meth = DynamicSystODE.solveODE_Num(y0 = y0_start(dyn_syst) , F = Fsimul , Ti = 0 , Tf = T_simul , h = hh, meth = num_meth)

                # Integration with UA method with ML modification
                Y_app_meth = DynamicSystODE.solveODE_Num(y0 = y0_start(dyn_syst) , F = Fhat , Ti = 0 , Tf = T_simul , h = hh , meth = num_meth)
                #TT = np.arange(0,T_simul,hh)
                #for n in range(np.size(TT)):
                #    Y_app_meth[:, n] = DynamicSyst.Phi(TT[n] / eeps, Y_app_meth[:, n], eeps, n_iter)

                # Computation of the error between the exact solution and the numerical solution ploted with f and the numerical method chosen
                err_f = np.array([np.linalg.norm((Y_exact - Y_exact_meth)[:, i]) for i in range((Y_exact - Y_exact_meth).shape[1])])
                Err_f = np.linalg.norm(err_f, np.infty) / norm_sol
                ERR_f[i,j] = Err_f
                #print("h = ",format(hh,'.4E')," - error = " , format(Err_f,'.4E'))

                # Computation of the error between the exact solution and the numerical solution ploted with f and the numerical method chosen with ML modification
                err_meth = np.array([np.linalg.norm((Y_exact - Y_app_meth)[:, i]) for i in range((Y_exact - Y_app_meth).shape[1])])
                Err_meth = np.linalg.norm(err_meth, np.infty) / norm_sol
                ERR_meth[i,j] = Err_meth

        if num_meth == "Forward Euler" or num_meth == "Integral_Euler":
            plt.figure()
            plt.title("Error between trajectories with " + num_meth + " - T = "+str(T_simul) + " - eps = "+str(eps_simul))
            for i in range(np.size(EEPS)):
                cmap = plt.get_cmap("hsv")
                L_eps = np.size(EEPS)
                Colors = [cmap(i / L_eps) for i in range(L_eps)]
                plt.plot(HH, ERR_f[i,:], marker="s" , linestyle="dashed" , color="red" , label = "Integral Euler")
                plt.plot(HH, ERR_meth[i, :], marker="s" , linestyle="dashed" , color="green", label = "Integral Euler + ML")
            # if len(ERR_f) > 0:
            #     plt.scatter(HH_f, ERR_f, label=num_meth + " - $f$", marker="s", color="red")
            #     torch.save(HH_f,"Trajectories_Time_steps_f_RKmodif_Comparison_K_50000_Nh_5")
            #     torch.save(ERR_f, "Trajectories_Errors_f_RKmodif_Comparison_K_50000_Nh_5")
            # if len(ERR_meth) > 0:
            #     plt.scatter(HH_meth, ERR_meth, label=num_meth + " - $f_{app}$", marker="s", color="green")
            #     if save_list == True:
            #         torch.save(HH_meth, "Trajectories_Time_steps_f_app_RKmodif_Comparison_K_50000_Nh_5")
            #         torch.save(ERR_meth, "Trajectories_Errors_f_app_RKmodif_Comparison_K_50000_Nh_5")
            # plt.scatter(HH, ERR_star_meth, label="$|\phi^{f,RK45}_{nh}(y_0) - (\Phi^{f_{app}^*}_{h})^n(y_0)|$", marker="s",color="orange")
            plt.legend()
            plt.grid()
            plt.xscale('log')
            plt.yscale('log')
            plt.xlabel("Time step")
            plt.ylabel("Global error")
            if save_fig == True:
                plt.savefig(name + dtime.now().strftime(' - %Y-%m-%d %H:%M:%S') + '.pdf', dpi=(200))
            plt.show()

        else:
            plt.figure()
            plt.title("Error between trajectories " + num_meth)
            if len(ERR_f) > 0:
                plt.scatter(HH_f, ERR_f, label=num_meth + " - $f$", marker="s", color="red")
            if len(ERR_meth) > 0:
                plt.scatter(HH_meth, ERR_meth, label=num_meth + " - $f_{app}$", marker="s", color="green")
                if save_list == True:
                    torch.save(HH_meth, "Trajectories_Time_steps_f_app_RKmodif_Comparison_K_50000_Nh_5")
                    torch.save(ERR_meth, "Trajectories_Errors_f_app_RKmodif_Comparison_K_50000_Nh_5")
            plt.legend()
            plt.grid()
            plt.xscale('log')
            plt.yscale('log')
            plt.xlabel("Time step")
            plt.ylabel("Global error")
            if save_fig == True:
                plt.savefig(name + dtime.now().strftime(' - %Y-%m-%d %H:%M:%S') + '.pdf', dpi=(200))
            plt.show()

        pass


class TimeCompute(Integrate):
    """Prints computational times for all methods used"""
    def time(self, model, name, save_fig):
        """Prints the computational times according to the step of the numerical method
        Inputs:
        - model: Best model learned during training
        - name: Character string - Potential name of the graph
        - save_fig: Boolean - Saves or not the figure"""

        Model = model[0]

        HH = np.exp(np.linspace(np.log(step_h[0]), np.log(step_h[1]), 11))
        #EEPS = np.exp(np.linspace(np.log(step_eps[0]), np.log(step_eps[1]), 1))
        EEPS = np.exp(np.linspace(np.log(eps_simul), np.log(eps_simul), 1))
        ERR_f1 , ERR_f2 , ERR_meth = np.zeros((np.size(EEPS),np.size(HH))) , np.zeros((np.size(EEPS),np.size(HH))) , np.zeros((np.size(EEPS),np.size(HH))) # Global errors
        TIME_f1, TIME_f2 , TIME_meth = np.zeros((np.size(EEPS), np.size(HH))), np.zeros((np.size(EEPS), np.size(HH))) , np.zeros((np.size(EEPS),np.size(HH)))  # Computational times

        for i in range(np.size(EEPS)):
            eeps = EEPS[i]
            for j in range(np.size(HH)):
                hh = HH[j]
                print(" - h = {} \r".format(format(hh, '.4E')),"eps = "+format(format(eeps, '.4E')), end="")

                def Fhat(t, y):
                    """Vector fied learned with the neural network, adapted for ODE integration
                    Inputs:
                    - t: Float - Time
                    - y: Array of shape (d,) - Space variable"""
                    y = torch.tensor(y).reshape(d, 1)
                    y.requires_grad = True
                    t_tensor = torch.tensor([[t]]).float()
                    h_tensor = torch.tensor([[hh]]).float()
                    eps_tensor = torch.tensor([[eeps]]).float()
                    z = Model(t_tensor/eps_tensor, y, eps_tensor, h_tensor)
                    # z = Model(y , h_tensor)
                    z = z.detach().numpy()
                    z = np.array(z, dtype=np.float64)
                    return z.reshape(d, )

                def Fsimul(t, y):
                    """Vector fied of the ODE, adapted for ODE integration
                    Inputs:
                    - t: Float - Time
                    - y: Array of shape (d,) - Space variable"""
                    if num_meth == "Forward Euler":
                        return DynamicSyst.f(t, y, eps_simul)
                    if num_meth == "Integral_Euler":
                        yy = hh * DynamicSyst.f_avg(y)
                        xi, omega = npp.polynomial.legendre.leggauss(10)
                        tau_0 = t / eps_simul + 2 * np.pi * np.round(hh / (2 * np.pi * eps_simul))
                        tau_1 = (t + hh) / eps_simul
                        xi = [((tau_1 - tau_0) / 2) * ksi + (tau_1 + tau_0) / 2 for ksi in xi]
                        for i in range(10):
                            yy += eps_simul * ((tau_1 - tau_0) / 2) * omega[i] * (DynamicSyst.f(eps_simul * xi[i], y, eps_simul) - DynamicSyst.f_avg(y))
                        return yy
                    if num_meth == "Integral_Euler_order2":
                        yy = hh * DynamicSyst.f_avg(y)
                        xi, omega = npp.polynomial.legendre.leggauss(10)
                        tau_0 = t / eps_simul + 2 * np.pi * np.round(hh / (2 * np.pi * eps_simul))
                        tau_1 = (t + hh) / eps_simul
                        xi1 = [((tau_1 - tau_0) / 2) * ksi + (tau_1 + tau_0) / 2 for ksi in xi]
                        for i in range(10):
                            yy += eps_simul * ((tau_1 - tau_0) / 2) * omega[i] * (DynamicSyst.f(eps_simul * xi1[i], y, eps_simul) - DynamicSyst.f_avg(y))

                        xi2 = [(h / 2) * ksi + t + h / 2 for ksi in xi]
                        def I(u , y):
                            """Second integral involved in the second order Euler Integral scheme.
                            Input:
                            - u: Float - Time variable
                            - y: Array of shape (d,) - Space variable"""

                            xxi2 = [((u-t) / 2) * ksi + (t + u) / 2 for ksi in xi]
                            II = 0*y

                            for i in range(10):
                                II += ((u-t)/2) * DynamicSyst.g_1(t , xxi2[i] , y , eps)
                            return II

                        for i in range(10):
                            yy += (h / 2)*I(xi2[i] , y)

                        return yy

                def Fsimul_Integral_Euler_order1(t, y):
                    """Vector fied of the ODE, adapted for ODE integration with Integral Euler scheme at order 1
                    Inputs:
                    - t: Float - Time
                    - y: Array of shape (d,) - Space variable"""
                    yy = hh * DynamicSyst.f_avg(y)
                    xi, omega = npp.polynomial.legendre.leggauss(10)
                    tau_0 = t / eps_simul + 2 * np.pi * np.round(hh / (2 * np.pi * eps_simul))
                    tau_1 = (t + hh) / eps_simul
                    xi = [((tau_1 - tau_0) / 2) * ksi + (tau_1 + tau_0) / 2 for ksi in xi]
                    for i in range(10):
                        yy += eps_simul * ((tau_1 - tau_0) / 2) * omega[i] * (DynamicSyst.f(eps_simul * xi[i], y, eps_simul) - DynamicSyst.f_avg(y))
                    return yy

                def Fsimul_Integral_Euler_order2(t, y):
                    """Vector fied of the ODE, adapted for ODE integration with Integral Euler scheme at order 2
                    Inputs:
                    - t: Float - Time
                    - y: Array of shape (d,) - Space variable"""
                    yy = hh * DynamicSyst.f_avg(y)
                    xi, omega = npp.polynomial.legendre.leggauss(10)
                    tau_0 = t / eps_simul + 2 * np.pi * np.round(hh / (2 * np.pi * eps_simul))
                    tau_1 = (t + hh) / eps_simul
                    xi1 = [((tau_1 - tau_0) / 2) * ksi + (tau_1 + tau_0) / 2 for ksi in xi]
                    for i in range(10):
                        yy += eps_simul * ((tau_1 - tau_0) / 2) * omega[i] * (DynamicSyst.f(eps_simul * xi1[i], y, eps_simul) - DynamicSyst.f_avg(y))

                    xi2 = [(hh / 2) * ksi + t + hh / 2 for ksi in xi]

                    def I(u , y):
                        """Second integral involved in the second order Euler Integral scheme.
                        Input:
                        - u: Float - Time variable
                        - y: Array of shape (d,) - Space variable"""

                        xxi2 = [((u-t) / 2) * ksi + (t + u) / 2 for ksi in xi]
                        II = 0*y

                        for i in range(10):
                            II += ((u-t)/2) * omega[i] * DynamicSyst.g_1(t , xxi2[i] , y , eps_simul)
                        return II

                    for i in range(10):
                        yy += (hh / 2) * omega[i] * I(xi2[i] , y)

                    return yy

                # Integration with DOP853 (approximation of the exact flow)
                #Y_exact = DynamicSystODE.solveODE(y0 = y0_start(dyn_syst) , T = T_simul , h = hh , eps = eeps , rel_tol = 1e-15, abs_tol = 1e-15)
                Y_exact = DynamicSystODE.solveODE(y0 = y0_start(dyn_syst) , Ti = 0 , Tf = T_simul , h = hh , eps = eeps)
                norm_sol = np.linalg.norm(np.array([np.linalg.norm((Y_exact)[:, i]) for i in range((Y_exact).shape[1])]),np.infty)  # Norm of the exact solution

                # Integration with UA method - order 1
                start_time_f1 = time.time()
                Y_exact_f1 = DynamicSystODE.solveODE_Num(y0 = y0_start(dyn_syst) , F = Fsimul_Integral_Euler_order1 , Ti = 0 , Tf = T_simul , h = hh, meth = num_meth)
                c_time_f1 = time.time() - start_time_f1
                TIME_f1[i,j] = c_time_f1
                err_f1 = np.array([np.linalg.norm((Y_exact - Y_exact_f1)[:, i]) for i in range((Y_exact - Y_exact_f1).shape[1])])
                Err_f1 = np.linalg.norm(err_f1, np.infty) / norm_sol
                ERR_f1[i, j] = Err_f1

                # Integration with UA method - order 1
                start_time_f2 = time.time()
                Y_exact_f2 = DynamicSystODE.solveODE_Num(y0=y0_start(dyn_syst), F=Fsimul_Integral_Euler_order2, Ti=0, Tf=T_simul, h=hh, meth=num_meth)
                c_time_f2 = time.time() - start_time_f2
                TIME_f2[i, j] = c_time_f2
                err_f2 = np.array([np.linalg.norm((Y_exact - Y_exact_f2)[:, i]) for i in range((Y_exact - Y_exact_f2).shape[1])])
                Err_f2 = np.linalg.norm(err_f2, np.infty) / norm_sol
                ERR_f2[i, j] = Err_f2

                # Integration with UA method with ML modification
                start_time_meth = time.time()
                Y_app_meth = DynamicSystODE.solveODE_Num(y0 = y0_start(dyn_syst) , F = Fhat , Ti = 0 , Tf = T_simul , h = hh , meth = num_meth)
                c_time_meth = time.time() - start_time_meth
                TIME_meth[i, j] = c_time_meth
                err_meth = np.array([np.linalg.norm((Y_exact - Y_app_meth)[:, i]) for i in range((Y_exact - Y_app_meth).shape[1])])
                Err_meth = np.linalg.norm(err_meth, np.infty) / norm_sol
                ERR_meth[i,j] = Err_meth

        if num_meth == "Forward Euler" or num_meth == "Integral_Euler":
            plt.figure()
            plt.title("Error vs computational time with " + num_meth + " - T = "+str(T_simul) + " - eps = "+str(eps_simul))
            for i in range(np.size(EEPS)):
                cmap = plt.get_cmap("hsv")
                L_eps = np.size(EEPS)
                Colors = [cmap(i / L_eps) for i in range(L_eps)]
                #plt.scatter(HH, ERR_f[i,:], label=num_meth + " - $f$", marker="s", color="red")

                #plt.scatter(HH, ERR_meth[i, :], label=num_meth + " - $f_app$", marker="s", color="green")
                plt.plot(TIME_f1[i, :], ERR_f1[i, :], marker="s", linestyle="dashed", color="red" , label = "Integral Euler - Order 1")
                plt.plot(TIME_f2[i, :], ERR_f2[i, :], marker="s", linestyle="dashed", color="orange" , label = "Integral Euler - Order 2")
                plt.plot(TIME_meth[i,:], ERR_meth[i, :], marker="s" , linestyle="dashed" , color="green", label = "Integral Euler + ML")
            # if len(ERR_f) > 0:
            #     plt.scatter(HH_f, ERR_f, label=num_meth + " - $f$", marker="s", color="red")
            #     torch.save(HH_f,"Trajectories_Time_steps_f_RKmodif_Comparison_K_50000_Nh_5")
            #     torch.save(ERR_f, "Trajectories_Errors_f_RKmodif_Comparison_K_50000_Nh_5")
            # if len(ERR_meth) > 0:
            #     plt.scatter(HH_meth, ERR_meth, label=num_meth + " - $f_{app}$", marker="s", color="green")
            #     if save_list == True:
            #         torch.save(HH_meth, "Trajectories_Time_steps_f_app_RKmodif_Comparison_K_50000_Nh_5")
            #         torch.save(ERR_meth, "Trajectories_Errors_f_app_RKmodif_Comparison_K_50000_Nh_5")
            # plt.scatter(HH, ERR_star_meth, label="$|\phi^{f,RK45}_{nh}(y_0) - (\Phi^{f_{app}^*}_{h})^n(y_0)|$", marker="s",color="orange")
            plt.legend()
            plt.grid()
            plt.xscale('log')
            plt.yscale('log')
            plt.xlabel("Computational time (s)")
            plt.ylabel("Global error")
            if save_fig == True:
                plt.savefig(name + dtime.now().strftime(' - %Y-%m-%d %H:%M:%S') + '.pdf', dpi=(200))
            plt.show()

        else:
            plt.figure()
            plt.title("Error between trajectories " + num_meth)
            if len(ERR_f) > 0:
                plt.scatter(HH_f, ERR_f, label=num_meth + " - $f$", marker="s", color="red")
            if len(ERR_meth) > 0:
                plt.scatter(HH_meth, ERR_meth, label=num_meth + " - $f_{app}$", marker="s", color="green")
                if save_list == True:
                    torch.save(HH_meth, "Trajectories_Time_steps_f_app_RKmodif_Comparison_K_50000_Nh_5")
                    torch.save(ERR_meth, "Trajectories_Errors_f_app_RKmodif_Comparison_K_50000_Nh_5")
            plt.legend()
            plt.grid()
            plt.xscale('log')
            plt.yscale('log')
            plt.xlabel("Time step")
            plt.ylabel("Global error")
            if save_fig == True:
                plt.savefig(name + dtime.now().strftime(' - %Y-%m-%d %H:%M:%S') + '.pdf', dpi=(200))
            plt.show()

        pass


def ExTest():
    """Test of Euler integral method for Logistic equation"""

    def G(tau , W , eps):
        """Micro-Macro vector field.
        Inputs:
        - tau: Float - Time variable
        - W: Array of shape (2d,) - Space variable
        - eps: Float - High oscillation parameter
        Returns an array of shape (2d,)"""

        v , w = W[0:d] , W[d:2*d]

        vv = v*(1-v)
        ww = (v+w)*(1-v-w) - v*(1-v) + np.array([np.sin(tau)])
        WW = np.concatenate( (vv,ww) , axis=0)


        return WW

    def G_av(W , eps):
        """Average Micro-Macro vector field for tensors.
        Inputs (K is the number of data):
        - W: Tensor of shape (2d,K) - Space variable
        - eps: Tensor of shape (1,K) - High oscillation parameter
        Returns an array of shape (2d,K)"""

        v, w = W[0:d], W[d:2 * d]

        vv = v * (1 - v)
        ww = (v + w) * (1 - v - w) - v * (1 - v)
        WW = np.concatenate((vv, ww), axis=0)

        return WW

    def G_av_NN(W,eps):
        v, w = W[0:d, :], W[d:2 * d, :]

        vv = v * (1 - v)
        ww = (v + w) * (1 - v - w) - v * (1 - v)
        WW = torch.cat((vv, ww), dim=0)
        return WW

    def RG_NN(tau , W , eps):
        """Zero - Average Micro-Macro vector field for tensors (G - G_av).
        Inputs (K is the number of data):
        - tau: Tensor of shape (1,K) - Time variable
        - W: Tensor of shape (2d,K) - Space variable
        - eps: Tensor of shape (1,K) - High oscillation parameter
        Returns an array of shape (2d,K)"""

        v, w = W[0:d, :], W[d:2 * d, :]

        vv = 0 * v
        ww = torch.tensor([[torch.sin(tau)]])
        WW = torch.cat((vv, ww), dim=0)
        return WW

    HH = torch.exp(torch.tensor(np.linspace(np.log(step_h[0]),np.log(step_h[1]),5)))
    ERR = []

    def dynamics(t,W):
        return G_av(W,1)

    for hh in HH:
        print("h=",format(hh,'.4E'))
        Sol_ex = solve_ivp(fun=dynamics , t_span=(0,T_simul) , y0=np.array([1.5,0]) , t_eval=np.arange(0,T_simul,hh) , method="DOP853" , rtol = 1e-10 , atol = 1e-10).y
        Sol_ex = torch.tensor(Sol_ex,dtype=torch.float32)

        Sol_app = torch.zeros_like(Sol_ex)
        Sol_app[:,0] = torch.tensor([1.5,0])
        W = torch.tensor([1.5,0])

        for n in range(Sol_app.shape[1]-1):
            t_n = n*hh
            tau_0 = n*hh + torch.round(hh/(2*np.pi))*2*np.pi
            tau_1 = (n+1)*hh

            xi_1, xi_2, xi_3, xi_4, xi_5, xi_6, xi_7, xi_8, xi_9, xi_10 = -0.1488743389816312, 0.1488743389816312, -0.4333953941292472, 0.4333953941292472, -0.6794095682990244, 0.6794095682990244, -0.8650633666889845, 0.8650633666889845, -0.9739065285171717, 0.9739065285171717
            omega_1, omega_2, omega_3, omega_4, omega_5, omega_6, omega_7, omega_8, omega_9, omega_10 = 0.2955242247147529, 0.2955242247147529, 0.2692667193099963, 0.2692667193099963, 0.2190863625159820, 0.2190863625159820, 0.1494513491505806, 0.1494513491505806, 0.0666713443086881, 0.0666713443086881
            tau_01, tau_02, tau_03, tau_04, tau_05, tau_06, tau_07, tau_08, tau_09, tau_010 = (tau_1 - tau_0) / 2 * xi_1 + (tau_0 + tau_1) / 2, (tau_1 - tau_0) / 2 * xi_2 + (tau_0 + tau_1) / 2, (tau_1 - tau_0) / 2 * xi_3 + (tau_0 + tau_1) / 2, (tau_1 - tau_0) / 2 * xi_4 + (tau_0 + tau_1) / 2, (tau_1 - tau_0) / 2 * xi_5 + (tau_0 + tau_1) / 2, (tau_1 - tau_0) / 2 * xi_6 + (tau_0 + tau_1) / 2, (tau_1 - tau_0) / 2 * xi_7 + (tau_0 + tau_1) / 2, (tau_1 - tau_0) / 2 * xi_8 + (tau_0 + tau_1) / 2, (tau_1 - tau_0) / 2 * xi_9 + (tau_0 + tau_1) / 2, (tau_1 - tau_0) / 2 * xi_10 + (tau_0 + tau_1) / 2



            W = Sol_app[:,n].reshape(2*d,1)

            Sol_app[:,n+1] = Sol_app[:,n] + hh*G_av_NN(W , 1).reshape(2*d,)# + (1/2)*(tau_1-tau_0)*(omega_1*RG_NN(tau_01,W,1) + omega_2*RG_NN(tau_02,W,1) + omega_3*RG_NN(tau_03,W,1) +omega_4*RG_NN(tau_04,W,1) +omega_5*RG_NN(tau_05,W,1) +omega_6*RG_NN(tau_06,W,1) +omega_7*RG_NN(tau_07,W,1) +omega_8*RG_NN(tau_08,W,1) +omega_9*RG_NN(tau_09,W,1) +omega_10*RG_NN(tau_010,W,1)).reshape(2*d,)

        print(Sol_app.dtype)
        diff = np.array(Sol_ex - Sol_app)
        err_f = np.array([np.linalg.norm((diff)[:, i]) for i in range((diff).shape[1])])
        Err_f = np.linalg.norm(err_f, np.infty)
        ERR = ERR + [Err_f]

    plt.figure()
    plt.loglog(HH,ERR,"sr",label="App")
    plt.grid()
    plt.legend()
    plt.show()

    pass




def ExData(name_data="DataODE_NA"):
    """Creates data y0, y1 with the function solvefEDOData
    with the chosen vector field at the beginning of the program
    Input:
    - name_data: Character string - Name of the registered tuple containing the data (default: "DataEDO_Traj")"""
    DataEDO = DynamicSystODE.solveODE_Data(K=K_data, p=p_train, h_data=step_h)
    torch.save(DataEDO, name_data)
    pass

def ExTrain(name_model="model_NA", name_data="DataODE_NA"):
    """Launches training and computes Loss_train, loss_test and best_model with the function Train().train
    Saves the files Loss_train, Loss_test and best_model with a given name
    Inputs (character strings):
    - name_model: Name of the file saved for best_model (default: "model_PB")
    - name_data: Name of the file containing the created data (default: "DataODE_PB") used for training"""
    DataODE = torch.load(name_data)
    Loss_train, Loss_test, best_model = Train().train(model=NN(), Data=DataODE)
    torch.save((best_model,Loss_train,Loss_test), name_model)
    pass

def ExIntegrate(name_model="model_NA", name_graph="Simulation_NA", save=False):
    """Launches integration of the main equation and modified equation with the chosen model
    Inputs:
    - name_model: Character string - Name of the model made with neural network chosen for integration with F_app, and Loss_train/Loss_test
    - name_graph: Character string - Name of the graph which will be registered
    - save: Boolean - Saves the figure or not (default: False)"""
    Lmodel = torch.load(name_model)
    Integrate().integrate(model=Lmodel, name=name_graph,save_fig=save)
    pass

def ExTraj(name_model="model_NA", name_graph="Simulation_Convergence_Trajectories", save=False):
    """plots the curves of convergence between the trajectories integrated with f and f_app with the numerical method chosen
    Inputs:
    - name_model: Character string - Name of the model made with neural network chosen for integration with f_app
    - name_graph: Character string - Name of the graph which will be registered
    - save: Boolean - Saves the figure or not (default: False)"""
    best_model = torch.load(name_model)
    Trajectories().traj(model=best_model, name=name_graph, save_fig=save)
    pass

def ExTime(name_model="model_NA", name_graph="Simulation_Convergence_Time", save=False):
    """plots the curves of convergence between the trajectories integrated with f and f_app with the numerical method chosen
    Inputs:
    - name_model: Character string - Name of the model made with neural network chosen for integration with f_app
    - name_graph: Character string - Name of the graph which will be registered
    - save: Boolean - Saves the figure or not (default: False)"""
    best_model = torch.load(name_model)
    TimeCompute().time(model=best_model, name=name_graph, save_fig=save)
    pass

def ExId_Data(name_data="DataODE_PB"):
    """Gives informations about data.
    Inputs:
     - name_data: Str - Name of the data set"""
    Data = torch.load(name_data)
    print(50*"-")
    print("Informations for data set - "+name_data)
    print(50*"-")
    print("  - Number of data:",Data[0].shape[1]+Data[1].shape[1])
    print("  - Train proportion:", 100*np.round(Data[0].shape[1]/(Data[0].shape[1]+Data[1].shape[1]),1),"%")
    print("  - Values for h:",[format(np.float(torch.min(Data[4])),'.2E'),format(np.float(torch.max(Data[5])),'.2E')])
    print("  - Values for eps:", [format(np.float(torch.min(Data[6])),'.2E'),format(np.float(torch.max(Data[6])),'.2E')])
    pass

def ExDataCombine(name_list,name_global):
    """Function to combine data sets created at the same time (parallel)
    Inputs:
    - name_list: List - Names of data sets
    - name_global: Str - Name of the saved global data set"""
    Data = torch.load(name_list[0])
    Data = list(Data)
    for k in range(len(name_list)-1):
        data = torch.load(name_list[k+1])
        for i in range(8):
            Data[i] = torch.cat((Data[i],data[i]),axis = 1)
    torch.save(tuple(Data),name_global)
