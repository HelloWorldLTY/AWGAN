import numpy as np
import pandas as pd
import graphtools as gt
import os
import datetime
import scanpy as sc
from skmisc.loess import loess
import torch.autograd
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.utils.data as Data  
from torchvision.utils import save_image
import numpy as np
import os
import pandas as pd
import torch.optim.lr_scheduler as lr_s 
from scipy.spatial.distance import cdist

from numba import jit

@jit(nopython=True) #utilzie numba acceleration
def pdist(vec1,vec2):
    """Function utilized to calculate cosine similarity between two vectors, which is called by the old correlation-finding function. 

    Args:
        vec1: vector1 with numpy format 
        vec2: vector2 with numpy format

    Output:
        Float: The cosine similarity of two given vectors. 

    """
  return vec1@vec2/(np.linalg.norm(vec1)*np.linalg.norm(vec2))

@jit(nopython=True)
def find_correlation_index_old(frame1, frame2):
    """Function utilized to find the index of NNPs existing in two single-cell RNA sequence profiles (Old version with smaller memory usage but slower running time).

    Args:
        frame1: count matrix 1 after pre-processing 
        frame2: count matrix 2 after pre-processing

    Output:
        result(list): A list containing index pair for NNPs.
    """
  result=[(1,1) for _ in range(len(frame2))]
  for i in range(len(frame2)):
    max_dist = -10 # or -inf
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


def find_correlation_index(frame1, frame2):
    """Function utilized to find the index of NNPs existing in two single-cell RNA sequence profiles (New version with faster running time but larger memory usage).

    Args:
        frame1: count matrix 1 after pre-processing 
        frame2: count matrix 2 after pre-processing

    Output:
        result(list): A list containing index pair for NNPs.
    """
  distlist =  cdist(frame2,frame1,metric='cosine')
  result = np.argmin(distlist,axis=1)
  result1 = []
  for i in range(len(frame2)):
    result1.append((i,result[i]))
  return result1

def training_set_generator(frame1,frame2,ref,batch):
    """Function utilized to find the index of NNPs existing in two single-cell RNA sequence profiles (New version with faster running time but larger memory usage).

    Args:
        frame1: count matrix 1 after pre-processing 
        frame2: count matrix 2 after pre-processing

    Output:
        result(list): A list containing index pair for NNPs.
    """
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
  """A class implementation for the state-of-the-art activation function, Mish.
  """
  def __init__(self):
    super().__init__()

  def forward(self,x):
    return x*torch.tanh(F.softplus(x))


class discriminator(nn.Module):
  """The discrimimator structure of our AWGAN. 
    Layer: 2000->1024->512->256->128->1
  """

    def __init__(self):
        super(discriminator, self).__init__()
        self.dis = nn.Sequential(
            nn.Linear(2000, 1024),  
            Mish(),
            nn.Linear(1024, 512),  
            Mish(),
            nn.Linear(512, 256),  
            Mish(),
            nn.Linear(256, 128),  
            Mish(),
            nn.Linear(128, 1)

        )

    def forward(self, x):
        x = self.dis(x)
        return x
 
 
# WGAN generator
# Require batch normalization
class generator(nn.Module):
  """The generator structure of our AWGAN. 
    Layer: 2000->1024->512->256->512->1024->2000
  """
    def __init__(self, drop_out=0.5):
        super(generator, self).__init__()
        self.relu_f = nn.ReLU(True)
        self.gen = nn.Sequential(
            nn.Linear(2000, 1024),
            nn.Dropout(drop_out),
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
            nn.Dropout(drop_out)
        )

    def forward(self, x):
        gre = self.gen(x)
        return self.relu_f(gre+x)    #residual network
 
def calculate_gradient_penalty(real_data, fake_data, D, center=1, p=2): 
    """Function utilized to calculate gradient penalty. This term is used to construct the loss function. 
    Args:
       real_data: Tensor of reference (new reference) batch data
       fake_data: Tensor of query batch data
       D: The discriminator network 
       center: K of Lipschitz condition 
       p: Dimensions of distance 

    Output:
        result(list): A list containing index pair for NNPs.
    """
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
  grad_penalty = ((gradients.norm(2, dim=1) - center) ** p).mean() 
  return grad_penalty 


@jit(nopython = True)
def determine_batch(val1,val_list =[32,64,128,256]):
  """Function utilized to determine suitable batch size
    Args:
       val1: Input batch size
       val_list: A list for candidiate batch size

    Output:
        result(list): A value for suitable batch size to avoid errors in batch correction.
  """
  for i in val_list:
    if val1%i !=1:
      return i
    else:
      continue
  return val1

def WGAN_train_type1(train_label,train_data,epoch,batch,lambda_1,val_list=[32,64,128,256]):
  """Function utilized to train WGAN. We call it as type1 because we utilize raw fake data to calculate the penalty.
    Args:
       train_label: Tensor of reference (new reference) batch data
       train_data: Tensor of query batch data
       epoch: The number of iteraiton steps
       batch: Input batch 
       lambda_1: A hyperparameter used to control the weights of gradient penalty
       val_list: A list for candidiate batch size 

    Output:
        final_list: Results for query batch after batch correction
        G: The generator of AWGAN
  """
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
  batch = determine_batch(MAX_ITER,val_list)

  train_label = torch.FloatTensor(train_label)
  train_data = torch.FloatTensor(train_data)
  if torch.cuda.is_available():
    train_label = train_label.cuda()
    train_data = train_data.cuda()
        

  for epoch_1 in range(epoch):
    print("This is ", epoch_1)
    for time in range(0,MAX_ITER,batch):
      true_data = train_label[time:time+batch,:]
      false_data = train_data[time:time+batch,:]
    

      #train d at first

      d_optimizer.zero_grad()

      real_out = D(true_data)
      real_label_loss = -torch.mean(real_out)

      # train use WGAN

      fake_out_new = G(false_data).detach()
      fake_out = D(fake_out_new)

      div = calculate_gradient_penalty(true_data, false_data, D)

      label_loss = real_label_loss+torch.mean(fake_out)+div/lambda_1
      label_loss.backward()



      d_optimizer.step()

  
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
  test_data = train_data
  if torch.cuda.is_available():
    final_list = G(test_data).detach().cpu().numpy()    
  else:
    final_list = G(test_data).detach().numpy()
  return final_list,G

def WGAN_train_type2(train_label,train_data,epoch,batch,lambda_1):
  """Function utilized to train WGAN. We call it as type2 because we utilize temporary correction data to calculate the penalty.
    Args:
       train_label: Tensor of reference (new reference) batch data
       train_data: Tensor of query batch data
       epoch: The number of iteraiton steps
       batch: Input batch 
       lambda_1: A hyperparameter used to control the weights of gradient penalty
       val_list: A list for candidiate batch size 

    Output:
        final_list: Results for query batch after batch correction
        G: The generator of AWGAN
  """
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

  train_label = torch.FloatTensor(train_label)
  train_data = torch.FloatTensor(train_data)
  if torch.cuda.is_available():
    train_label = train_label.cuda()
    train_data = train_data.cuda()
        

  for epoch_1 in range(epoch):
    print("This is ", epoch_1)
    for time in range(0,MAX_ITER,batch):
      true_data = train_label[time:time+batch,:]
      false_data = train_data[time:time+batch,:]
    

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
  test_data = train_data
  if torch.cuda.is_available():
    final_list = G(test_data).detach().cpu().numpy()    
  else:
    final_list = G(test_data).detach().numpy()    
  return final_list,G

def sequencing_train(ref_adata, batch_adata, batch_inf, epoch=100, batch=32, lambda_1=1/10, type_key=1):
  """Function utilized to train WGAN. We call it as type2 because we utilize temporary correction data to calculate the penalty.
    Args:
       ref_adata: Reference batch data 
       batch_adata: Query batches data (including >=1 batch(es))
       batch_inf: Batch index
       epoch: The number of iteraiton steps
       batch: Input batch 
       lambda_1: A hyperparameter used to control the weights of gradient penalty
       type_key:  AWGAN training type 

    Output:
        ref_data_ori: Final results after batch correction, for the whole dataset
        G_tar: Final generator
  """
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