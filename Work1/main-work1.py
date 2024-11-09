import torch
import torch.nn as nn
import numpy as np
import os
import time
import datetime
import argparse
import scipy.io as io
import math
from torch import optim, autograd
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from torch.utils.data import Dataset, DataLoader
from DataSets  import Sample_Point, Exact_Solution
from Utils import helper

from Models.FcNet import FcNet

from itertools import cycle


print("pytorch version", torch.__version__, "\n")

## parser arguments
parser = argparse.ArgumentParser(description='Deep Residual Method for discontinuous solution')
# checkpoints
parser.add_argument('-c', '--checkpoint', default='Checkpoints/Burgers/simulation_3', type=str, metavar='PATH', help='path to save checkpoint')
# figures 
parser.add_argument('-i', '--image', default='Images/Burgers/simulation_3', type=str, metavar='PATH', help='path to save figures')
parser.add_argument('-r', '--result', default='Results/Burgers/simulation_3', type=str, metavar='PATH', help='path to sabe results')
parser.add_argument('--num_epochs', default=15000, type=int, metavar='N', help='number of total epochs to run')
args = parser.parse_args()
##############################################################################################



##################################################################################################



# dataset setting

parser.add_argument('--beta', default=400, type=int, metavar='N', help='penalty coefficeint for mismatching of boundary data')
parser.add_argument('--milestones', type=int, nargs='+', default=[5000, 8000, 12000, 14000], help='decrease learning rate at these epochs')
parser.add_argument('--num_batches', default=5, type=int, metavar='N',help='number of mini-batches during training')

# network architecture
parser.add_argument('--depth', type=int, default=2, help='network depth')
parser.add_argument('--width', type=int, default=40, help='network width')

# datasets options
parser.add_argument('--num_intrr_pts', type=int, default=80000, help='total number of interior sampling points')
parser.add_argument('--num_init_pts', type=int, default=5000, help='total number of sampling points of initial points')
parser.add_argument('--num_bndry_pts', type = int, default=5000, help='number of sampling points of boundary')
parser.add_argument('--num_test_t', type=int, default=101, help='number of time jet')
parser.add_argument('--num_test_x', type=int, default=101, help='number of sampling points at a certain time')
parser.add_argument('--num_shock_pts', type=int, default=5000, help='number of sampling points at characteristic lines')

args = parser.parse_known_args()[0]

# problem setting
dim_prob = 2

batchsize_intrr_pts = args.num_intrr_pts // args.num_batches
batchsize_init_pts = 2 * args.num_init_pts // args.num_batches
batchsize_bndry_pts = 2 * args.num_bndry_pts // args.num_batches
batchsize_shock_pts = 2 * args.num_shock_pts // args.num_batches
################################################################################################

################################################################################################
print('*', '-' * 45, '*')
print('===> preparing training and testing datasets ...')
print('*', '-' * 45, '*')

# training dataset for sample points inside the domain
class TraindataInterior(Dataset):    
    def __init__(self, num_intrr_pts, dim_prob): 
        
        self.SmpPts_Interior = Sample_Point.SmpPts_Interior(num_intrr_pts, dim_prob)        
               
    def __len__(self):
        return len(self.SmpPts_Interior)
    
    def __getitem__(self, idx):
        SmpPt = self.SmpPts_Interior[idx]

        return [SmpPt]

# training dataset for sample points at the Dirichlet boundary
class TraindataBoundary(Dataset):    
    def __init__(self, num_bndry_pts, dim_prob):         
        
        self.SmpPts_Bndryl, self.SmpPts_Bndryr = Sample_Point.SmpPts_Boundary(num_bndry_pts, dim_prob)
        
    def __len__(self):
        return len(self.SmpPts_Bndryl)
    
    def __getitem__(self, idx):
        SmpPtl = self.SmpPts_Bndryl[idx]
        SmpPtr = self.SmpPts_Bndryr[idx]

        return [SmpPtl, SmpPtr]    
    
class TraindataInitial(Dataset):
    def __init__(self, num_init_pts, dim_prob):

        self.SmpPts_Init = Sample_Point.SmpPts_Initial(num_init_pts, dim_prob)
        self.f_Exact_SmpPts = Exact_Solution.f_Exact(self.SmpPts_Init[:,0])

    def __len__(self):
        return len(self.SmpPts_Init)
    
    def __getitem__(self, idx):
        SmpPts_Init = self.SmpPts_Init[idx]
        f_SmpPts = self.f_Exact_SmpPts[idx]

        return [SmpPts_Init, f_SmpPts]
    
class TraindataShock(Dataset):
    def __init__(self, num_shock_pts, dim_prob):
        self.SmpPts_Shock1, self.SmpPts_Shock2 = Sample_Point.SmpPts_Shock(num_shock_pts, dim_prob)
    
    def __len__(self):
        return len(self.SmpPts_Shock1)
    
    def __getitem__(self, idx):
        SmpPts_Shock1 = self.SmpPts_Shock1[idx]
        SmpPts_Shock2 = self.SmpPts_Shock2[idx]

        return [SmpPts_Shock1, SmpPts_Shock2]


class Testdata(Dataset):
    def __init__(self, num_test_t, num_test_x):
        self.SmpPts_Test = Sample_Point.SmpPts_Test(num_test_x, num_test_t)
        self.u_Exact_Solution = Exact_Solution.u_Exact_Solution(self.SmpPts_Test[:,0], self.SmpPts_Test[:,1])

    def __len__(self):
        return len(self.SmpPts_Test)
    
    def __getitem__(self, idx):
        SmpPts = self.SmpPts_Test[idx]
        u_Exact_SmpPts = self.u_Exact_Solution[idx]

        return [SmpPts, u_Exact_SmpPts]
    

################################################################################################



################################################################################################
# create training and testing datasets         
traindata_intrr = TraindataInterior(args.num_intrr_pts, dim_prob)
traindata_bndry = TraindataBoundary(args.num_bndry_pts, dim_prob)
traindata_init = TraindataInitial(args.num_init_pts, dim_prob)
traindata_shock = TraindataShock(args.num_shock_pts, dim_prob)
testdata = Testdata(args.num_test_t, args.num_test_x)

# define dataloader
dataloader_intrr = DataLoader(traindata_intrr, batch_size=batchsize_intrr_pts, shuffle=True, num_workers=0)
dataloader_bndry = DataLoader(traindata_bndry, batch_size=batchsize_bndry_pts, shuffle=True, num_workers=0)
dataloader_init = DataLoader(traindata_init, batch_size=batchsize_init_pts, shuffle=True, num_workers=0)
dataloader_shock = DataLoader(traindata_shock, batch_size=batchsize_shock_pts, shuffle=True, num_workers=0)
dataloader_test = DataLoader(testdata, batch_size=args.num_test_t*args.num_test_x, shuffle=True, num_workers=0)
####################################################################################################

##############################################################################################
# plot sample points during training and testing
if not os.path.isdir(args.image):
    helper.mkdir_p(args.image)

fig = plt.figure()
plt.scatter(traindata_intrr.SmpPts_Interior[:,0], traindata_intrr.SmpPts_Interior[:,1], c = 'red', label = 'interior points' )
plt.scatter(traindata_bndry.SmpPts_Bndryl[:,0], traindata_bndry.SmpPts_Bndryl[:,1], c = 'blue', label = 'boundry points' )
plt.scatter(traindata_bndry.SmpPts_Bndryr[:,0], traindata_bndry.SmpPts_Bndryr[:,1], c = 'blue', label = 'boundry points' )
plt.scatter(traindata_init.SmpPts_Init[:,0], traindata_init.SmpPts_Init[:,1], c = 'yellow', label = 'initial points')
plt.scatter(traindata_shock.SmpPts_Shock1[:,0], traindata_shock.SmpPts_Shock1[:,1], c = 'pink', label = 'characteristic line')
plt.scatter(traindata_shock.SmpPts_Shock2[:,0], traindata_shock.SmpPts_Shock2[:,1], c = 'pink', label = 'characteristic line')
plt.title('Sample Points during Training')
plt.legend(loc = 'lower right')
# plt.show()
plt.savefig(os.path.join(args.image,'TrainSmpPts.png'))
plt.close(fig)

fig = plt.figure()
plt.scatter(testdata.SmpPts_Test[:,0], testdata.SmpPts_Test[:,1], c = 'black')
plt.title('Sample Points during Testing')
# plt.show()
plt.savefig(os.path.join(args.image,'TestSmpPts.png'))
plt.close(fig)
##############################################################################################

##############################################################################################
print('*', '-' * 45, '*')
print('===> creating training model ...')
print('*', '-' * 45, '*', "\n", "\n")

def train_epoch(epoch, model, optimizer, device):
    
    # set model to training mode
    model.train()

    loss_epoch, loss_intrr_epoch, loss_bndry_epoch, loss_init_epoch, loss_shock_epoch = 0, 0, 0, 0, 0

    # ideally, sample points within the interior domain and at its boundary have the same number of mini-batches
    # otherwise, it wont's shuffle the dataloader_boundary samples again when it starts again (see https://discuss.pytorch.org/t/two-dataloaders-from-two-different-datasets-within-the-same-loop/87766/7)

    for i, (data_intrr, data_bndry, data_init, data_shock) in enumerate(zip(dataloader_intrr, cycle(dataloader_bndry), cycle(dataloader_init), cycle(dataloader_shock))):

        # get mini-batch training data
        [smppts_intrr] = data_intrr
        smppts_bndryl, smppts_bndryr = data_bndry
        smppts_init, f_smppts = data_init
        smppts_shock1, smppts_shock2= data_shock

        # add the third variable
        smppts_intrr = torch.cat([smppts_intrr, Exact_Solution.level_set(smppts_intrr[:,0], smppts_intrr[:,1]).reshape(-1,1)], dim=1)
        smppts_bndryl = torch.cat([smppts_bndryl, Exact_Solution.level_set(smppts_bndryl[:,0], smppts_bndryl[:,1]).reshape(-1,1)], dim=1)
        smppts_bndryr = torch.cat([smppts_bndryr, Exact_Solution.level_set(smppts_bndryr[:,0], smppts_bndryr[:,1]).reshape(-1,1)], dim=1)
        smppts_init = torch.cat([smppts_init, Exact_Solution.level_set(smppts_init[:,0], smppts_init[:,1]).reshape(-1,1)], dim=1)



        smppts_shock1l = torch.cat([smppts_shock1, Exact_Solution.level_set(smppts_shock1[:,0] - 0.0001, smppts_shock1[:,1]).reshape(-1,1)], dim=1)
        smppts_shock1r = torch.cat([smppts_shock1, Exact_Solution.level_set(smppts_shock1[:,0] + 0.0001, smppts_shock1[:,1]).reshape(-1,1)], dim=1)

        smppts_shock2l = torch.cat([smppts_shock2, Exact_Solution.level_set(smppts_shock2[:,0] - 0.0001, smppts_shock2[:,1]).reshape(-1,1)], dim=1)
        smppts_shock2r = torch.cat([smppts_shock2, Exact_Solution.level_set(smppts_shock2[:,0] + 0.0001, smppts_shock2[:,1]).reshape(-1,1)], dim=1)
    

        smppts_intrr = smppts_intrr.to(device)
        f_smppts = f_smppts.to(device)
        smppts_bndryl = smppts_bndryl.to(device)
        smppts_bndryr = smppts_bndryr.to(device)
        smppts_init = smppts_init.to(device)

        smppts_shock1l = smppts_shock1l.to(device)
        smppts_shock1r = smppts_shock1r.to(device)
        smppts_shock2l = smppts_shock2l.to(device)
        smppts_shock2r = smppts_shock2r.to(device)

        Prosp1 = 1/4 *  torch.ones(smppts_shock1l.size(0)).reshape(-1,1)
        Prosp1 = Prosp1.to(device)

        Prosp2 = 1 / (2 * torch.sqrt(smppts_shock2[:, 1]))
        Prosp2 = Prosp2.to(device)

        smppts_intrr.requires_grad = True

        # forward pass to obtain NN prediction of u(x)
        u_NN_intrr = model(smppts_intrr)
        u_NN_bndryl = model(smppts_bndryl)
        u_NN_bndryr = model(smppts_bndryr)
        u_NN_init = model(smppts_init)

        u_NN_shock1l = model(smppts_shock1l)
        u_NN_shock1r = model(smppts_shock1r)
        u_NN_shock2l = model(smppts_shock2l)
        u_NN_shock2r = model(smppts_shock2r)

        

        # zero parameter gradients and then compute NN prediction of gradient u(x)
        model.zero_grad()
        gradu_NN_intrr = torch.autograd.grad(outputs=u_NN_intrr, inputs=smppts_intrr, grad_outputs=torch.ones_like(u_NN_intrr), retain_graph=True, create_graph=True, only_inputs=True)[0]

        # construct mini-batch loss function and then perform backward pass
        loss_intrr = torch.mean(torch.pow(gradu_NN_intrr[:,1] +  0.5 *  torch.squeeze(u_NN_intrr) * gradu_NN_intrr[:,0], 2))
        loss_bndry = torch.mean(torch.pow(u_NN_bndryl, 2)) + torch.mean(torch.pow(u_NN_bndryr, 2))
        loss_init = torch.mean(torch.pow(torch.squeeze(u_NN_init) - f_smppts, 2))


        loss_shock = torch.mean(torch.pow(torch.squeeze(u_NN_shock1r + u_NN_shock1l) / 4 - Prosp1, 2)) + torch.mean(torch.pow(torch.squeeze(u_NN_shock2r + u_NN_shock2l) / 4 - Prosp2, 2))

        # if epoch >= 5000:
        #     alpha = 200
        # else:
        #     alpha = 1
        alpha = 300

        loss_minibatch = alpha * loss_intrr + loss_bndry + args.beta * (loss_init + loss_shock)

        #zero parameter gradients
        optimizer.zero_grad()
        # backpropagation
        loss_minibatch.backward()
        # parameter update
        optimizer.step()

        # integrate loss over the entire training dataset
        loss_intrr_epoch += loss_intrr.item() * smppts_intrr.size(0) / traindata_intrr.SmpPts_Interior.shape[0]
        loss_bndry_epoch += loss_bndry.item() * smppts_bndryl.size(0) / traindata_bndry.SmpPts_Bndryl.shape[0]
        loss_init_epoch += loss_init.item() * smppts_init.size(0) / traindata_init.SmpPts_Init.shape[0]
        loss_shock_epoch += loss_shock.item() * smppts_shock1.size(0) / traindata_shock.SmpPts_Shock1.shape[0]
        loss_epoch += alpha * loss_intrr_epoch + loss_bndry_epoch + args.beta * (loss_init_epoch + loss_shock_epoch) 
        
    return loss_intrr_epoch, loss_bndry_epoch, loss_init_epoch, loss_shock_epoch, loss_epoch
        

##############################################################################################
print('*', '-' * 45, '*')
print('===> creating testing model ...')
print('*', '-' * 45, '*', "\n", "\n")

def test_epoch(epoch, model, optimizer, device):
    
    # set model to testing mode
    model.eval()

    epoch_loss_u= 0
    for smppts_test, u_exact_smppts in dataloader_test:
        
        # send inputs, outputs to device.
        smppts_test = torch.cat([smppts_test, Exact_Solution.level_set(smppts_test[:,0], smppts_test[:,1]).reshape(-1,1)], dim=1)
        smppts_test = smppts_test.to(device)
        u_exact_smppts = u_exact_smppts.to(device)  
        
        smppts_test.requires_grad = True
        
        # forward pass and then compute loss function for approximating u by u_NN
        u_NN_smppts = model(smppts_test) 
        
        loss_u = torch.mean(torch.pow(torch.squeeze(u_NN_smppts) - u_exact_smppts, 2))         
        
        # integrate loss      
        epoch_loss_u += loss_u.item()          
    
    return epoch_loss_u
################################################################################################


##############################################################################################
print('*', '-' * 45, '*')
print('===> neural network training ...')

if not os.path.isdir(args.checkpoint):
    helper.mkdir_p(args.checkpoint)

# create model
model = FcNet.FcNet(dim_prob + 1,args.width,1,args.depth)
model.Xavier_initi()
print('Network Architecture:', "\n", model)
print('Total number of trainable parameters = ', sum(p.numel() for p in model.parameters() if p.requires_grad))

# create optimizer and learning rate schedular
optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01, amsgrad=False)
schedular = torch.optim.lr_scheduler.MultiStepLR(optimizer, args.milestones, gamma=0.1)

# load model to device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('DEVICE: {}'.format(device), "\n")
model = model.to(device)

# create log file
logger = helper.Logger(os.path.join(args.checkpoint, 'log.txt'), title='Deep-Rtiz-Poisson-Square2D')
logger.set_names(['Learning Rate', 'Train Loss', 'Test Loss', 'TrainLoss Interior', 'TrainLoss Bndry', 'Trainloss Init', 'Trainloss Shock'])
     
# train and test 
train_loss, test_loss_u = [], []
trainloss_best = 1e10
since = time.time()
for epoch in range(args.num_epochs):
      
    print('Epoch {}/{}'.format(epoch, args.num_epochs-1), 'with LR = {:.1e}'.format(optimizer.param_groups[0]['lr']))  
        
    # execute training and testing
    trainloss_intrr_epoch, trainloss_bndry_epoch, trainloss_init_epoch, trainloss_shock_epoch, trainloss_epoch = train_epoch(epoch, model, optimizer, device)
    testloss_u_epoch = test_epoch(epoch, model, optimizer, device)
    
    # save current and best models to checkpoint
    is_best = trainloss_epoch < trainloss_best
    if is_best:
        print('==> Saving best model ...')
    trainloss_best = min(trainloss_epoch, trainloss_best)
    helper.save_checkpoint({'epoch': epoch + 1,
                            'state_dict': model.state_dict(),
                            'trainloss_intrr_epoch': trainloss_intrr_epoch,
                            'trainloss_bndry_epoch': trainloss_bndry_epoch,
                            'trainloss_init_epoch': trainloss_init_epoch,
                            'trainloss_shock_epoch': trainloss_shock_epoch,
                            'trainloss_epoch': trainloss_epoch,
                            'testloss_u_epoch': testloss_u_epoch,
                            'trainloss_best': trainloss_best,
                            'optimizer': optimizer.state_dict(),
                           }, is_best, checkpoint=args.checkpoint)   
    # save training process to log file
    logger.append([optimizer.param_groups[0]['lr'], trainloss_epoch, testloss_u_epoch, trainloss_intrr_epoch, trainloss_bndry_epoch, trainloss_init_epoch, trainloss_shock_epoch])
    
    # adjust learning rate according to predefined schedule
    schedular.step()

    # print results
    train_loss.append(trainloss_epoch)
    test_loss_u.append(testloss_u_epoch)
    print('==> Full-Batch Training Loss = {:.4e}'.format(trainloss_epoch))
    print('    Fubb-Batch Testing Loss : ', 'u-u_NN = {:.4e}'.format(testloss_u_epoch), "\n")
    
logger.close()
time_elapsed = time.time() - since

# # save learning curves
# helper.save_learncurve({'train_curve': train_loss, 'test_curve': test_loss}, curve=args.image)  

print('Done in {}'.format(str(datetime.timedelta(seconds=time_elapsed))), '!')
print('*', '-' * 45, '*', "\n", "\n")
##############################################################################################

##############################################################################################
# plot learning curves
fig = plt.figure()
plt.plot(torch.log10(torch.tensor(train_loss)), c = 'red', label = 'training loss' )
plt.title('Learning Curve during Training')
plt.legend(loc = 'upper right')
# plt.show()
fig.savefig(os.path.join(args.image,'TrainCurve.png'))

fig = plt.figure()
plt.plot(torch.log10(torch.tensor(test_loss_u)), c = 'red', label = 'testing loss (u)' )
plt.title('Learning Curve during Testing')
plt.legend(loc = 'upper right')
# plt.show()
fig.savefig(os.path.join(args.image,'TestCurve.png'))
##############################################################################################

##############################################################################################
print('*', '-' * 45, '*')
print('===> loading trained model for inference ...')

# load trained model
checkpoint = torch.load(os.path.join(args.checkpoint, 'model_best.pth.tar'))
model.load_state_dict(checkpoint['state_dict'])

# compute NN predicution of u and gradu
with torch.no_grad():  
    t = torch.linspace(0, 1, steps = args.num_test_t) * 10
    temp = torch.ones(args.num_test_x, 1) 
    init = torch.zeros(args.num_test_x, 1)
    test_smppts = torch.cat([torch.linspace(0, 1, steps=args.num_test_x).reshape(-1, 1) * 7 - 1, temp.reshape(-1,1)], dim=1)
    test_smppts = torch.cat([test_smppts, Exact_Solution.level_set(test_smppts[:,0], test_smppts[:,1]).reshape(-1,1)], dim=1)
    test_smppts_t2 = torch.cat([torch.linspace(0, 1, steps=args.num_test_x).reshape(-1, 1) * 7 - 1, 6 * temp.reshape(-1,1)], dim=1)
    test_smppts_t2 = torch.cat([test_smppts_t2, Exact_Solution.level_set(test_smppts_t2[:,0], test_smppts_t2[:,1]).reshape(-1,1)], dim=1)
    init_smppts = torch.cat([torch.linspace(0, 1, steps=args.num_test_x).reshape(-1, 1) * 7 - 1, init.reshape(-1,1)], dim=1)
    init_smppts = torch.cat([init_smppts, Exact_Solution.level_set(init_smppts[:,0], init_smppts[:,1]).reshape(-1,1)], dim=1)


    test_smppts = test_smppts.to(device)
    init_smppts = init_smppts.to(device)
    test_smppts_t2 = test_smppts_t2.to(device)
    u_NN = model(test_smppts)
    u_NN_2 = model(test_smppts_t2)
    u_init = model(init_smppts)

x = torch.squeeze(test_smppts[:,0]).cpu()

xs, ts = torch.meshgrid(x, t)
test_intrr = torch.squeeze(torch.stack([xs.reshape(1, args.num_test_t * args.num_test_x), ts.reshape(1, args.num_test_t * args.num_test_x)], dim=-1))
test_intrr = torch.cat([test_intrr, Exact_Solution.level_set(test_intrr[:, 0], test_intrr[:, 1]).reshape(-1, 1)], dim=1)

test_intrr = test_intrr.to(device)
u_test = model(test_intrr).cpu().detach()

test_exact = Exact_Solution.u_Exact_Solution(test_intrr[:, 0].cpu(), test_intrr[:, 1].cpu()).detach()

fig = plt.figure()
plt.contourf(xs, ts, test_exact.reshape(xs.size()), 40, cmap='jet')
plt.title('Exact Solution u on TestData')
plt.colorbar()
fig.savefig(os.path.join(args.image, 'Exact_u'))
plt.close(fig)

x = torch.squeeze(test_smppts[:,0]).cpu().detach().numpy().reshape(args.num_test_x, 1)
    
# plot u and its network prediction on testing dataset
fig=plt.figure()
u_Exact = Exact_Solution.u_Exact_Solution(test_smppts[:,0].cpu(),test_smppts[:,1].cpu()).detach().numpy().reshape(args.num_test_x, 1)
plt.plot(x, u_Exact, ls = '-', lw = '2')
plt.title('Exact Solution u on Test Dataset')
#plt.show()  
fig.savefig(os.path.join(args.image,'Exact_u_testdata_N.png'))
plt.close(fig)


fig=plt.figure()
u_NN = u_NN.cpu().detach().numpy().reshape(args.num_test_x, 1)
plt.plot(x, u_NN, ls='-')
plt.title('Predicted u_NN on Test_data')
#plt.show()
fig.savefig(os.path.join(args.image, 'predited_u_testdata_N.png'))
plt.close(fig)

fig=plt.figure()
u_NN_2 = u_NN_2.cpu().detach().numpy().reshape(args.num_test_x, 1)
plt.plot(x, u_NN_2, ls='-')
plt.title('Predicted u_NN on Test_data')
#plt.show()
fig.savefig(os.path.join(args.image, 'predited_u_testdata_N_t2.png'))
plt.close(fig)

fig = plt.figure()
u_init = u_init.cpu().detach().numpy().reshape(args.num_test_x, 1)
plt.plot(x, u_init, ls='-')
plt.title('Predicted u_init on initial condition')
fig.savefig(os.path.join(args.image, 'predited_u_init.png'))
plt.close(fig)

fig = plt.figure()
plt.contourf(xs, ts, u_test.reshape(xs.size()), 40, cmap='jet')
plt.title('Predicted u in entire domain')
plt.colorbar()
fig.savefig(os.path.join(args.image, 'predited_u_NN'))
plt.close(fig)

print('*', '-' * 45, '*', "\n", "\n")

if not os.path.isdir(args.result):
    helper.mkdir_p(args.result)

torch.save(model.state_dict(), args.result+'/model_weight.pt')
##############################################################################################


                                           


