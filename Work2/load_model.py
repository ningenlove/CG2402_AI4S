'''This is a sample file to illustrate how to load model and generate data that can be used for matlab plotting'''

import torch
import argparse
from Models.FcNet import FcNet
import os
import scipy.io as io
from matplotlib import pyplot as plt
import numpy as np

from DataSets import Sample_Point, Exact_Solution
from Utils import helper

import h5py

## parser arguments
parser = argparse.ArgumentParser(description='Deep Residual Method for discontinuous solution')
parser.add_argument('-c', '--checkpoint', default='Checkpoints/Burgers/simulation_0', type=str, metavar='PATH', help='path to save checkpoint')
parser.add_argument('-i', '--image', default='Images/simulation_0', type=str, metavar='PATH', help='path to save figures')
parser.add_argument('-r', '--result', default='Results/simulation_0', type=str, metavar='PATH', help='path to sabe results')
args = parser.parse_args()
##############################################################################################

##################################################################################################
# network architecture
parser.add_argument('--depth', type=int, default=2, help='network depth')
parser.add_argument('--width', type=int, default=80, help='network width')
parser.add_argument('--num_test_t', type=int, default=1001, help='number of time jet')
parser.add_argument('--num_test_x', type=int, default=1001, help='number of sampling points at a certain time')
args = parser.parse_known_args()[0]
##############################################################################################

dim_prob = 2

model = FcNet.FcNet(dim_prob + 1,args.width,1,args.depth)
checkpoint = torch.load(os.path.join(args.checkpoint, 'model_best.pth.tar'), map_location=torch.device('cpu'))
model.load_state_dict(checkpoint['state_dict'])

## Generate the path to save images

if not os.path.isdir(args.image):
    helper.mkdir_p(args.image)

## Generate the path to save result
if not os.path.isdir(args.result):
    helper.mkdir_p(args.result)



########################################################################
xs = torch.linspace(-1, 6, steps = args.num_test_x)
ts = torch.linspace(0, 10, steps = args.num_test_t)
x, t = torch.meshgrid(xs, ts)

c = torch.ones(x.size())

#######################################################################
# c1 = 0.1
test_smppts_c1 = torch.squeeze(torch.stack([x.reshape(1, args.num_test_t * args.num_test_x), (0.1 * c * t / 11).reshape(1, args.num_test_t * args.num_test_x)], dim=-1))
test_smppts_c1 = torch.cat([test_smppts_c1, Exact_Solution.level_set(test_smppts_c1[:,0], test_smppts_c1[:,1]).reshape(-1,1)], dim=1)

with torch.no_grad():
    u_pred_c1 = model(test_smppts_c1)

u_pred_c1 = u_pred_c1.cpu().detach().numpy().reshape(-1,1)
u_Exact_c1 = Exact_Solution.u_Exact_Solution(test_smppts_c1[:,0],test_smppts_c1[:,1]).cpu().detach().numpy().reshape(-1, 1)
u_err_c1 = u_Exact_c1 - u_pred_c1
io.savemat(args.result+"/u_NN_projected_c1.mat",{"u_pred": u_pred_c1})
io.savemat(args.result+"/pterr_c1.mat", {"u_error":u_err_c1})
io.savemat(args.result+"/u_exact_c1.mat",{"u_exact":u_Exact_c1})
########################################################################

#######################################################################
# c2 = 1
test_smppts_c2 = torch.squeeze(torch.stack([x.reshape(1, args.num_test_t * args.num_test_x), (1 * c * t / 11).reshape(1, args.num_test_t * args.num_test_x)], dim=-1))
test_smppts_c2 = torch.cat([test_smppts_c2, Exact_Solution.level_set(test_smppts_c2[:,0], test_smppts_c2[:,1]).reshape(-1,1)], dim=1)

with torch.no_grad():
    u_pred_c2 = model(test_smppts_c2)

u_pred_c2 = u_pred_c2.cpu().detach().numpy().reshape(-1,1)
u_Exact_c2 = Exact_Solution.u_Exact_Solution(test_smppts_c2[:,0],test_smppts_c2[:,1]).cpu().detach().numpy().reshape(-1, 1)
u_err_c2 = u_Exact_c2 - u_pred_c2
io.savemat(args.result+"/u_NN_projected_c2.mat",{"u_pred": u_pred_c2})
io.savemat(args.result+"/pterr_c2.mat", {"u_error":u_err_c2})
io.savemat(args.result+"/u_exact_c2.mat",{"u_exact":u_Exact_c2})
########################################################################

#######################################################################
# c3 = 6
test_smppts_c3 = torch.squeeze(torch.stack([x.reshape(1, args.num_test_t * args.num_test_x), (6 * c * t / 11).reshape(1, args.num_test_t * args.num_test_x)], dim=-1))
test_smppts_c3 = torch.cat([test_smppts_c3, Exact_Solution.level_set(test_smppts_c3[:,0], test_smppts_c3[:,1]).reshape(-1,1)], dim=1)

with torch.no_grad():
    u_pred_c3 = model(test_smppts_c3)

u_pred_c3 = u_pred_c3.cpu().detach().numpy().reshape(-1,1)
u_Exact_c3 = Exact_Solution.u_Exact_Solution(test_smppts_c3[:,0],test_smppts_c3[:,1]).cpu().detach().numpy().reshape(-1, 1)
u_err_c3 = u_Exact_c3 - u_pred_c3
io.savemat(args.result+"/u_NN_projected_c3.mat",{"u_pred": u_pred_c3})
io.savemat(args.result+"/pterr_c3.mat", {"u_error":u_err_c3})
io.savemat(args.result+"/u_exact_c3.mat",{"u_exact":u_Exact_c3})
########################################################################
