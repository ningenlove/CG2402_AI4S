import torch
import math

def u_Exact_Solution(x, t):
    
    u = torch.zeros(x.size())

    # select the points at the domain of t < 1/c
    mask1 = t < 1/11
    x1 = x[mask1]
    t1 = t[mask1]

    # exact solution when t < 1/c
    u1 = torch.ones(x1.size())
    mask = 11 * t1 <= x1
    u1[mask] = (1 - x1[mask]) / (1 - 11 * t1[mask])
    mask = x1 >= 1
    u1[mask] = 0
    
    u[mask1] = u1

    # select the points at the domain of t >= 1/c
    mask2 = t >= 1/11
    x2 = x[mask2]
    t2 = t[mask2]

    # exact solution when t >= 1/c
    u2 = torch.ones(x2.size())
    mask = x2 >= 11/2 * t2 + 1/2
    u2[mask] =0

    u[mask2] = u2

    return u




def f_Exact(x):
    
    f = torch.ones(x.size(0))
    mask = x >= 0 
    f[mask] = 1 - x[mask]
    mask = x >= 1
    f[mask] = 0
    return f

    

def level_set(x, t):
    temp = torch.ones(x.size())
    level = torch.heaviside(x - 0.5 * 11 * t - 0.5, torch.squeeze(temp))
    return level


