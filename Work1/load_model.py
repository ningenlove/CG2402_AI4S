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
parser.add_argument('--width', type=int, default=40, help='network width')
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



temp = torch.ones(args.num_test_x, 1)
# generate test point when t1 = 2
test_smppts_t1 = torch.cat([torch.linspace(-1, 6, steps = args.num_test_x).reshape(-1, 1), 2 * temp.reshape(-1, 1)], dim = 1)
# lift the dimension
test_smppts_t1 = torch.cat([test_smppts_t1, Exact_Solution.level_set(test_smppts_t1[:, 0], test_smppts_t1[:, 1]).reshape(-1, 1)], dim = 1)

u_NN_t1 = model(test_smppts_t1).detach().numpy()
u_Exact_t1 = Exact_Solution.u_Exact_Solution(test_smppts_t1[:,0],test_smppts_t1[:,1]).numpy().reshape(args.num_test_x, 1)
###
# save the data for matlab plotting
io.savemat(args.result+"/u_NN_t1.mat",{"u_NN_t1": u_NN_t1})
io.savemat(args.result+"/u_exact_t1.mat", {"u_Exact_t1":u_Exact_t1})
io.savemat(args.result+"/pterr_t1.mat", {"pterr_t1":u_NN_t1 - u_Exact_t1})

# plot the figure at t1 = 2

x = torch.squeeze(test_smppts_t1[:,0]).numpy().reshape(args.num_test_x, 1)
fig=plt.figure()
plt.plot(x, u_Exact_t1, ls='-')
plt.title('Exact Solution u on Test Dataset')
#plt.show()  
plt.legend()
fig.savefig(os.path.join(args.image,'Exact_u_t1.png'))
plt.close(fig)

fig=plt.figure()
io.savemat(args.result+"/u_NN_t1.mat", {"u_NN_t1":u_NN_t1})
plt.plot(x, u_NN_t1, ls='-')
plt.title('Predicted u_NN on Test_data')
#plt.show()
fig.savefig(os.path.join(args.image, 'predited_u_NN_t1.png'))
plt.close(fig)

# generate test point when t2 = 4
test_smppts_t2 = torch.cat([torch.linspace(-1, 6, steps = args.num_test_x).reshape(-1, 1), 4 * temp.reshape(-1, 1)], dim = 1)
# lift the dimension
test_smppts_t2 = torch.cat([test_smppts_t2, Exact_Solution.level_set(test_smppts_t2[:, 0], test_smppts_t2[:, 1]).reshape(-1, 1)], dim = 1)

u_NN_t2 = model(test_smppts_t2).detach().numpy()
u_Exact_t2 = Exact_Solution.u_Exact_Solution(test_smppts_t2[:,0],test_smppts_t2[:,1]).numpy().reshape(args.num_test_x, 1)
###
# save the data for matlab plotting
io.savemat(args.result+"/u_NN_t2.mat",{"u_NN_t2": u_NN_t2})
io.savemat(args.result+"/u_exact_t2.mat", {"u_Exact_t2":u_Exact_t2})
io.savemat(args.result+"/pterr_t2.mat", {"pterr_t2":u_NN_t2 - u_Exact_t2})

# generate test point when t3 = 8
test_smppts_t3 = torch.cat([torch.linspace(-1, 6, steps = args.num_test_x).reshape(-1, 1), 8 * temp.reshape(-1, 1)], dim = 1)
# lift the dimension
test_smppts_t3 = torch.cat([test_smppts_t3, Exact_Solution.level_set(test_smppts_t3[:, 0], test_smppts_t3[:, 1]).reshape(-1, 1)], dim = 1)

u_NN_t3 = model(test_smppts_t3).detach().numpy()
u_Exact_t3 = Exact_Solution.u_Exact_Solution(test_smppts_t3[:,0],test_smppts_t3[:,1]).numpy().reshape(args.num_test_x, 1)
###
# save the data for matlab plotting
io.savemat(args.result+"/u_NN_t3.mat",{"u_NN_t3": u_NN_t3})
io.savemat(args.result+"/u_exact_t3.mat", {"u_Exact_t3":u_Exact_t3})
io.savemat(args.result+"/pterr_t3.mat", {"pterr_t3":u_NN_t3 - u_Exact_t3})
######################################################################


########################################################################
xs = torch.linspace(-1, 6, steps = args.num_test_x)
ts = torch.linspace(0, 10, steps = args.num_test_t)
x, t = torch.meshgrid(xs, ts)


test_smppts = torch.squeeze(torch.stack([x.reshape(1, args.num_test_t * args.num_test_x), t.reshape(1, args.num_test_t * args.num_test_x)], dim=-1))
temp = torch.ones(args.num_test_t * args.num_test_x)
for i in torch.arange(2):
    test_smppt_level = torch.cat([test_smppts, i * temp.reshape(-1,1)], dim =1)

    with torch.no_grad():
        u_level = model(test_smppt_level)

    u_level = u_level.cpu().detach().numpy().reshape(-1, 1)
    io.savemat(args.result+"/u_NN_level%d.mat"%i,{"u_NN_level":u_level})

########################################################################


########################################################################
test_smppts = torch.cat([test_smppts, Exact_Solution.level_set(test_smppts[:,0], test_smppts[:,1]).reshape(-1,1)], dim=1)

with torch.no_grad():
    u_pred = model(test_smppts)

u_pred = u_pred.cpu().detach().numpy().reshape(-1,1)
u_Exact = Exact_Solution.u_Exact_Solution(test_smppts[:,0],test_smppts[:,1]).cpu().detach().numpy().reshape(-1, 1)
u_err = u_Exact - u_pred
io.savemat(args.result+"/u_NN_projected.mat",{"u_pred": u_pred})
io.savemat(args.result+"/pterr.mat", {"u_error":u_err})
io.savemat(args.result+"/u_exact.mat",{"u_exact":u_Exact})
########################################################################


########################################################################
xs = torch.linspace(0, 1, steps=args.num_test_x) * 7 - 1
le = torch.linspace(0, 1, steps=args.num_test_x)
xt, l = torch.meshgrid(xs, le)
t1 = torch.ones(args.num_test_x * args.num_test_x, 1) * 2

t2 = torch.ones(args.num_test_x * args.num_test_x, 1) * 4

t3 = torch.ones(args.num_test_x * args.num_test_x, 1) * 8

pred_smppts_t1 = torch.cat([xt.reshape(-1, 1), t1.reshape(-1, 1)], dim=1)
pred_smppts_t1 = torch.cat([pred_smppts_t1, l.reshape(-1, 1)], dim=1)

pred_smppts_t2 = torch.cat([xt.reshape(-1, 1), t2.reshape(-1, 1)], dim=1)
pred_smppts_t2 = torch.cat([pred_smppts_t2, l.reshape(-1, 1)], dim=1)

pred_smppts_t3 = torch.cat([xt.reshape(-1, 1), t3.reshape(-1, 1)], dim=1)
pred_smppts_t3 = torch.cat([pred_smppts_t3, l.reshape(-1, 1)], dim=1)

u_NN_lif_t1 = model(pred_smppts_t1).cpu().detach().numpy().reshape(-1,1)
u_NN_lif_t2 = model(pred_smppts_t2).cpu().detach().numpy().reshape(-1,1)
u_NN_lif_t3 = model(pred_smppts_t3).cpu().detach().numpy().reshape(-1,1)


io.savemat(args.result+"/u_pred_t1.mat",{"u_pred_t1": u_NN_lif_t1})
io.savemat(args.result+"/u_pred_t2.mat",{"u_pred_t2": u_NN_lif_t2})
io.savemat(args.result+"/u_pred_t3.mat",{"u_pred_t3": u_NN_lif_t3})


#########################################################################


########################################################################

print("l^infty:", np.linalg.norm(u_err, np.inf))
print("related l^infty", np.linalg.norm(u_err, np.inf)/np.linalg.norm(u_Exact, np.inf))
print('l^2:',np.linalg.norm(u_err))
print('related l^2:',np.linalg.norm(u_err)/np.linalg.norm(u_Exact))

# mask = u_err > max(abs(u_err)) - 0.001
# print(test_smppts[:, 0].reshape(-1, 1)[mask])
# print(test_smppts[:, 1].reshape(-1, 1)[mask])