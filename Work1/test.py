import torch
import argparse
from Models.FcNet import FcNet
import os
import scipy.io as io
from matplotlib import pyplot as plt
import time
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
parser.add_argument('--width', type=int, default=40, help='network width')
parser.add_argument('--num_test_t', type=int, default=1001, help='number of time jet')
parser.add_argument('--num_test_x', type=int, default=1001, help='number of sampling points at a certain time')
args = parser.parse_known_args()[0]
##############################################################################################

dim_prob = 2

model = FcNet.FcNet(dim_prob + 1,args.width,1,args.depth)
checkpoint = torch.load(os.path.join(args.checkpoint, 'model_best.pth.tar'), map_location=torch.device('cpu'))
model.load_state_dict(checkpoint['state_dict'])


# Define a function whose input are the test points and output are predicted solution and test time
def get_solu(test_smppts): 
    test_smppts = torch.cat([test_smppts, Exact_Solution.level_set(test_smppts[:, 0], test_smppts[:, 1]).reshape(-1, 1)], dim=1)
    since = time.time()
    with torch.no_grad():
        u_NN = model(test_smppts)
    time_elapsed = time.time() - since

    return u_NN, time_elapsed

irre_file = h5py.File('Task1_test_1108_irreg.hdf5', 'r')
re_file = h5py.File('Task1_test_1108_reg.hdf5', 'r')

irre_u_exact = irre_file['u_data'][:].reshape(-1, 1)
irre_test_smppts = torch.tensor(irre_file['xt_points'])

re_u_exact = re_file['u_data'][:].reshape(-1, 1)
re_test_smppts = torch.tensor(re_file['xt_points'])


u_NN_irre, test_time_ir = get_solu(irre_test_smppts)
u_NN_re, test_time_re = get_solu(re_test_smppts)

u_NN_irre = u_NN_irre.cpu().detach().numpy().reshape(-1,1)
u_NN_re = u_NN_re.cpu().detach().numpy().reshape(-1,1)

irre_err = irre_u_exact - u_NN_irre
re_err = re_u_exact - u_NN_re

print('l^2 related error of irregular points:', np.linalg.norm(irre_err)/np.linalg.norm(irre_u_exact))
print('l^2 related error of regular points:', np.linalg.norm(re_err)/np.linalg.norm(re_u_exact))

