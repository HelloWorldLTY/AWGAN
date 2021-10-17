import scprep
import imap  #used for feature detected
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import phate
import graphtools as gt
import magic
import os
import datetime
import scanpy as sc
from skmisc.loess import loess
import sklearn.preprocessing as preprocessing

import umap.umap_ as umap

import torch.autograd
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torchvision import transforms
from torchvision import datasets
import torch.utils.data as Data  #Data是用来批训练的模块
from torchvision.utils import save_image
import numpy as np
import os
import pandas as pd
import torch.optim.lr_scheduler as lr_s 
from scipy.spatial.distance import cdist

#calculate cos distence
@jit(nopython=True)
def pdist(vec1,vec2):
  return vec1@vec2/(np.linalg.norm(vec1)*np.linalg.norm(vec2))

#calculate correlation index
#calculate cos distence
@jit(nopython=True)
def find_correlation_index(frame1, frame2):
  result=[(1,1) for _ in range(len(frame2))]
  for i in range(len(frame2)):
    max_dist = -10
    it1=0
    it2=0
    for j in range(len(frame1)):
      dist = pdist(frame2[i],frame1[j])
      if dist>max_dist:
        max_dist = dist
        it1 = i
        it2 = j 
    result[i] = (it1, it2)
  return result

#another method used for calculating correlation index
def find_correlation_index(frame1, frame2, size=3000):
  randomlist = np.array([i for i in range(len(frame1))])
  pick_list = np.random.choice(randomlist, size=size, replace=False)
  distlist =  cdist(frame2,frame1[pick_list],metric='cosine')
  result = np.argmin(distlist,axis=1)
  result1 = []
  for i in range(len(frame2)):
    result1.append((i,pick_list[result[i]]))
  return result1

def training_set_generator(frame1,frame2,ref,batch):
      common_pair = find_correlation_index(frame1,frame2)
  result = []
  result1 = []
  for i in common_pair:
    result.append(ref[i[1],:])
    result1.append(batch[i[0],:])
  return np.array(result),np.array(result1)


np.random.seed(999)
torch.manual_seed(999)
torch.cuda.manual_seed_all(999)

class Mish(nn.Module):
      def __init__(self):
    super().__init__()

  def forward(self,x):
    return x*torch.tanh(F.softplus(x))

#WGAN model, and it does not need to use bath normalization based on WGAN paper.
class discriminator(nn.Module):
    def __init__(self):
        super(discriminator, self).__init__()
        self.dis = nn.Sequential(
            nn.Linear(2000, 1024),  
            #nn.BatchNorm1d(1024),
            Mish(),
            nn.Linear(1024, 512),  
            #nn.BatchNorm1d(512),
            Mish(),
            nn.Linear(512, 256),  
            #nn.BatchNorm1d(256),
            Mish(),
            nn.Linear(256, 128),  
            #nn.BatchNorm1d(128),
            Mish(),
            nn.Linear(128, 1)

        )

    def forward(self, x):
        x = self.dis(x)
        return x
 
 
# WGAN generator
# Require batch normalization
class generator(nn.Module):
    def __init__(self):
        super(generator, self).__init__()
        self.relu_f = nn.ReLU(True)
        self.gen = nn.Sequential(
            nn.Linear(2000, 1024),
            nn.BatchNorm1d(1024),
            Mish(),

            nn.Linear(1024, 512),  
            nn.BatchNorm1d(512),
            Mish(),

            nn.Linear(512, 256),  
            nn.BatchNorm1d(256),
            Mish(),

            nn.Linear(256, 512),  
            nn.BatchNorm1d(512),
            Mish(),
  

            nn.Linear(512, 1024),  
            nn.BatchNorm1d(1024),
            Mish(),

            nn.Linear(1024, 2000),
            nn.Dropout(0.5)
        )

    def forward(self, x):
        gre = self.gen(x)
        return self.relu_f(gre+x)    #residual network
 
# calculate gradient penalty
def calculate_gradient_penalty(real_data, fake_data, D, center=1): 
  eta = torch.FloatTensor(real_data.size(0),1).uniform_(0,1) 
  eta = eta.expand(real_data.size(0), real_data.size(1)) 
  cuda = True if torch.cuda.is_available() else False 
  if cuda: 
    eta = eta.cuda() 
  else: 
    eta = eta 
  interpolated = eta * real_data + ((1 - eta) * fake_data) 
  if cuda: 
    interpolated = interpolated.cuda() 
  else: 
    interpolated = interpolated 
   # define it to calculate gradient 
  interpolated = Variable(interpolated, requires_grad=True) 
   # calculate probability of interpolated examples 
  prob_interpolated = D(interpolated) 
  # calculate gradients of probabilities with respect to examples 
  gradients = torch.autograd.grad(outputs=prob_interpolated, inputs=interpolated, 
  grad_outputs=torch.ones( 
  prob_interpolated.size()).cuda() if cuda else torch.ones( 
  prob_interpolated.size()), 
  create_graph=True, retain_graph=True)[0] 
  grad_penalty = ((gradients.norm(2, dim=1) - center) ** 2).mean() 
  return grad_penalty 

# parameters
EPOCH = 100
# MAX_ITER = train_data.shape[0]
batch = 4
b1 = 0.9
b2 = 0.999
lambda_1 = 1/10

@jit(nopython = True)
def determine_batch(val1):
  val_list =[32,64,128,256]
  for i in val_list:
    if val1%i !=1:
      return i
    else:
      continue
  return val1

def WGAN_train_type1(train_label,train_data,epoch,batch,lambda_1):
      stop = 0
  iter = 0
  D = discriminator()
  G = generator()

  if torch.cuda.is_available():
    D = D.cuda()
    G = G.cuda()
  lr=0.0001

  d_optimizer = torch.optim.AdamW(D.parameters(), lr=lr)
  g_optimizer = torch.optim.AdamW(G.parameters(), lr=lr)  

  G.train()
  D.train()

  MAX_ITER = train_data.shape[0]
  batch = determine_batch(MAX_ITER)

  for epoch_1 in range(epoch):
    print("This is ", epoch_1)
    for time in range(0,MAX_ITER,batch):
      true_data = torch.FloatTensor(train_label[time:time+batch,:]).cuda()
      false_data = torch.FloatTensor(train_data[time:time+batch,:]).cuda()
    

      #train d at first

      d_optimizer.zero_grad()

      real_out = D(true_data)
      real_label_loss = -torch.mean(real_out)

      # err_D.append(real_label_loss.cpu().float())

      # train use WGAN

      fake_out_new = G(false_data).detach()
      fake_out = D(fake_out_new)

      div = calculate_gradient_penalty(true_data, false_data, D)

      label_loss = real_label_loss+torch.mean(fake_out)+div/lambda_1
      label_loss.backward()

      # err_D.append(label_loss.cpu().item())

      d_optimizer.step()
      # scheduler_D.step()
  
      #train G

      real_out = G(false_data)
      real_output = D(real_out)

      real_loss1 = -torch.mean(real_output)
      # err_G.append(real_loss1.cpu().item())

      g_optimizer.zero_grad()

      real_loss1.backward()
      g_optimizer.step()
      # scheduler_G.step()

      if(time%100==0):
        print("g step loss",real_loss1)
      iter += 1
  print("Train step finished")
  G.eval()
  test_data = torch.FloatTensor(train_data).cuda()
  test_list = G(test_data).detach().cpu().numpy()    
  return test_list,G

def WGAN_train_type2(train_label,train_data,epoch,batch,lambda_1):
      stop = 0
  iter = 0
  D = discriminator()
  G = generator()

  if torch.cuda.is_available():
    D = D.cuda()
    G = G.cuda()
  lr=0.0001

  d_optimizer = torch.optim.AdamW(D.parameters(), lr=lr)
  g_optimizer = torch.optim.AdamW(G.parameters(), lr=lr)  

  G.train()
  D.train()

  MAX_ITER = train_data.shape[0]
  batch = determine_batch(MAX_ITER)

  for epoch_1 in range(epoch):
    print("This is ", epoch_1)
    for time in range(0,MAX_ITER,batch):
      true_data = torch.FloatTensor(train_label[time:time+batch,:]).cuda()
      false_data = torch.FloatTensor(train_data[time:time+batch,:]).cuda()
    

      #train d at first

      d_optimizer.zero_grad()

      real_out = D(true_data)
      real_label_loss = -torch.mean(real_out)


      # train use WGAN

      fake_out_new = G(false_data).detach()
      fake_out = D(fake_out_new)

      div = calculate_gradient_penalty(true_data, fake_out_new, D)

      label_loss = real_label_loss+torch.mean(fake_out)+div/lambda_1
      label_loss.backward()


      d_optimizer.step()

  
      #train G

      real_out = G(false_data)
      real_output = D(real_out)

      real_loss1 = -torch.mean(real_output)

      g_optimizer.zero_grad()

      real_loss1.backward()
      g_optimizer.step()


      if(time%100==0):
        print("g step loss",real_loss1)
      iter += 1
  print("Train step finished")
  G.eval()
  test_data = torch.FloatTensor(train_data).cuda()
  test_list = G(test_data).detach().cpu().numpy()    
  return test_list,G

def sequencing_train(ref_adata, batch_adata, batch_inf, epoch=100, batch=32, lambda_1=1/10, type_key=1):
      ref_data_ori = ref_adata.X
  for bat_inf in batch_inf[1:]:
    print("##########################Training%s#####################"%(bat_inf))
    batch_data_ori = batch_adata[batch_adata.obs['batch'] == bat_inf].X
    label_data,train_data = training_set_generator(ref_data_ori, batch_data_ori, ref_data_ori, batch_data_ori)
    print("#################Finish Pair finding##########################")
    if type_key==1:
      remove_batch_data,G_tar = WGAN_train_type1(label_data,train_data,epoch,batch,lambda_1)
    else:
      remove_batch_data,G_tar = WGAN_train_type2(label_data,train_data,epoch,batch,lambda_1)   
    ref_data_ori = np.vstack([ref_data_ori,remove_batch_data])
  print("###################### Finish Training ###########################")
  return ref_data_ori, G_tar