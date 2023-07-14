#!/usr/bin/env python
# coding: utf-8

# In[340]:


from IPython.display import display, HTML
display(HTML("<style>.container { width:100% !important; }</style>"))


import sys
import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import deeptime
import numpy as np
import matplotlib.pyplot as plt
#import pytorch_lightning as pl
import torch.nn.functional as F
from torch.utils.data import DataLoader,Dataset, TensorDataset
import gc
from contextlib import contextmanager
from tqdm import tqdm
from collections import OrderedDict
from torch.optim import Adam
from typing import Callable
from multiprocessing.pool import ThreadPool
import time
import pickle
from collections import ChainMap
from IPython.display import display
import scipy.stats as stats
import matplotlib.colors as colors
from itertools import repeat
#from torch.Nested import nested_tensor ###dreaming of pytorch to drop nested tensor support

plt.rcParams['axes.linewidth'] = 3

def num_str(s,return_str = True, return_num = True):
    s = ''.join(filter(str.isdigit, s))
    
    if return_str and return_num:
        return s,int(s)
    
    if return_str:
        return s
    
    if return_num:
        return int(s)
    
def make_symbols():
    unicharacters = ["\u03B1",
                        "\u03B2",
                        "\u03B3",
                        "\u03B4",
                        "\u03B5",
                        "\u03B6",
                        "\u03B7",
                        "\u03B8",
                        "\u03B9",
                        "\u03BA",
                        "\u03BB",
                        "\u03BC",
                        "\u03BD",
                        "\u03BE",
                        "\u03BF",
                        "\u03C0",
                        "\u03C1",
                        "\u03C2",
                        "\u03C3",
                        "\u03C4",
                        "\u03C5",
                        "\u03C6",
                        "\u03C7",
                        "\u03C8",
                        "\u03C9",
                        "\u00C5"]
    keys = "alpha,beta,gamma,delta,epsilon,zeta,eta,theta,iota,kappa,lambda,mu,nu,xi,omicron,pi,rho,final_sigma,sigma,tau,upsilon,phi,chi,psi,omega,angstrom"
    return dict(zip(keys.split(","),unicharacters))
symbols = make_symbols()


def make_trial_dir(trial_dir):
    """recursive function to create the next iteration in a series of indexed directories 
    to save training results generated from repeated trials. Guarantees correctly indexed
    directories and allows for lazy initialization (i.e finds the next index automatically
    assuming the "root" or "base" directory name remains the same)"""

    if not os.path.exists(trial_dir):
        os.makedirs(trial_dir)
        return trial_dir
    else:
        path_list = trial_dir.split("/")
        local_dir = path_list.pop()
        global_dir = "/".join(path_list)+"/"
        s,s_num = num_str(local_dir)
        trial_dir = global_dir + local_dir.replace(s,str(s_num+1))
        trial_dir = make_trial_dir(trial_dir)
        return trial_dir

def load_dict(file):
    with open(file,"rb") as handle:
        dic_loaded = pickle.load(handle)
    return dic_loaded
def save_dict(file, dict):
    with open(file, "wb") as handle:
        pickle.dump(dict, handle)
    return None

def check_mem():
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) or hasattr(obj, "data") and torch.is_tensor(obj.data):                print(type(obj), obj.size())
        except:
            pass



def pmf1d(x,nbins,range=None, weights = None, return_bin_centers = True):
    count,edge = np.histogram(x,bins = nbins,range=range,weights = weights)
    if weights is None:
        p = count/len(x)
    else:
        p = count
    if return_bin_centers:
        return p,edge[:-1]+np.diff(edge)/2
    else:
        return p


class timer:
    """import time"""
    def __init__(self,check_interval:"the time (hrs) after the call method should return True"):
        
        self.start_time = time.time()
        
        self.interval = check_interval*(60**2)
    
    def __call__(self):
        if abs(time.time() - self.start_time)>self.interval:
            self.start_time = time.time()
            return True
        else:
            return False
    
    def time_remaining(self):
        sec = max(0,self.interval - abs(time.time() - self.start_time))
        hrs = sec//(60**2)
        mins_remaining = (sec/60 - hrs*(60))
        mins = mins_remaining//1
        secs = (mins_remaining - mins)*60
        hrs,mins,secs = [int(i) for i in [hrs,mins,secs]]
        print(f"{hrs}:{mins}:{secs}")
        return None
        
    #for context managment
    def __enter__(self):
        self.start = time.time()
        return self 
    def __exit__(self, *args):
        self.end = time.time()
        self.interval = self.end - self.start
        print(f"Time elapsed {self.interval} s")
        return self.interval

def expectation(obs,p):
    """obs = time series vector or matrix with time in the zeroth dimension, features (or variables) in the second
    time series (0 dim) probability assignments to each state (1 dim)"""
    return (obs.T@p)/p.sum(0)

def expectation_sd(obs,p,exp = None):
    if exp is None:
        exp = expectation(obs, p)
    return np.sqrt(expectation(obs**2, p) - exp**2)


def pooled_sd(means:"1d array of trial means", sds:"1d array of trial sds",
              n_samples:"1d array of the number of samples used to estimate each sd and mean" = None,):
    """
    For combining standard deviations.
    
    
    Can be used for combining standard deviations estimated from datasets with differing number of samples.
    
    If n_samples if None or a constant, then it's assumed that the number of samples is the same for all SDs and cancels out of the sum and reduces to the number of standard deviations 
    being combined. As a result, this parameter can be left as None if all standard deviations are estimated using the same number of samples
    
    """
    if isinstance(n_samples, (float, int)) or n_samples is None:
        #in this case the number of samples cancels out
        return np.sqrt((sds**2 + (means - means.mean())**2).sum()/len(means))
    else:
        n = n_samples.sum()
        return np.sqrt((n_samples*(sds**2 + (means - means.mean())**2)).sum()/n)
    
def moving_average(x:np.ndarray, n:int=3) :
    ret = np.cumsum(x, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


# In[368]:


def plain_state_dict(d, badword = "module"):
    new = OrderedDict()
    for k,v in d.items():
        name = k.replace(badword+".", "")
        new[name] = v
    return new

def proj2d(p,c, alpha:"float or array of floats" = None, 
           state_map = False, ax = None, filename=None, cmap = "jet",
           comp_type:str = "Comp.", cbar_label:str = "Magnitude",
           x_label:str = None, y_label:str = None, title:str = None):
    if ax is None:
        fig,ax=plt.subplots(1,1,figsize=(5,5))
    if state_map:
        nstates = c.max()+1
        color_list = plt.cm.jet
        cs = [color_list(i) for i in range(color_list.N)]
        cmap = colors.ListedColormap(cs)
        boundaries = np.arange(nstates+1).tolist()
        norm = colors.BoundaryNorm(boundaries, cmap.N, clip=True)
        tick_locs = (np.arange(nstates) + 0.5)
        ticklabels = np.arange(1,nstates+1).astype(str).tolist()
        s = ax.scatter(p[:,0],p[:,1],c=c,s=.5,cmap=cmap,norm=norm)
        cbar = plt.colorbar(s,ax=ax)
        cbar.set_label("State",size=10)
        cbar.set_ticks(tick_locs)
        cbar.set_ticklabels(ticklabels)
    else:
        if (x_label is None and y_label is None):
            x_label,y_label = [f"{comp_type} {i}" for i in range(1,3)]
        s = ax.scatter(p[:,0],p[:,1],c=c,s=.5,cmap=cmap, alpha = alpha)
        cbar = plt.colorbar(s,ax=ax)
        cbar.set_label(cbar_label,size=15)
    if not title is None:
        ax.set_title(title, size = 20)
    ax.set_xlabel(x_label,fontsize=15)
    ax.set_ylabel(y_label,fontsize=15)
    ax.tick_params(axis="x",labelsize=10)
    ax.tick_params(axis="y",labelsize=10)
    cbar.ax.tick_params(labelsize=8)
    if not filename is None:
        plt.savefig(filename)
        plt.clf()
    return

def plot_free_energy(hist2d:np.ndarray, T:float,
                     title:str,xlabel:str,ylabel:str,
                     x_centers:np.ndarray,y_centers:np.ndarray):
        plt.figure(figsize=(10,10))
        free_energy= -T*0.001987*np.log(hist2d+.000001)        
        im=plt.imshow(free_energy, interpolation='gaussian',
                      extent=[x_centers[0], x_centers[-1],
                      y_centers[0], y_centers[-1]],
                      cmap='jet',aspect='auto')
        imaxes = plt.gca()
        plt.ylabel(ylabel,size=44,labelpad=15)
        plt.xlabel(xlabel,size=44,labelpad=15)
        plt.title(title,size=35)
        #HARD CODED TO Q IN THE X-AXIS
        plt.xticks(ticks = [.2,.4,.6,.8,1], labels = "0.2,0.4,0.6,0.8,1".split(","),fontsize=40)
        plt.yticks(fontsize=40)
        cbar_ticks=[0,1,2,3]
        cb = plt.colorbar(ticks=cbar_ticks, format=('% .1f'), aspect=10)
        cb.set_label("Free Energy / (kT)",size=45)
        cb.ax.tick_params(labelsize = 40)
        plt.axes(cb.ax)
        plt.clim(vmin=.1, vmax=3.0)
        plt.xlim(0,1)
        plt.tight_layout()




def reindex_dtraj(dtraj, obs, maximize_obs = True):
    """given a discrete trajectory and an observable, we reindex the trajectory 
    based on the mean of the observable in each state (high to low)
    maximize_obs has been added to increase flexibility as one might want to reindex
    states in order of smallest mean value of observable"""
    nstates = dtraj.max()+1 #number of states
    #get the sorted cluster indices based on mean of observable
    if maximize_obs:
        idx = np.array([obs[np.where(dtraj == i)[0]].mean() for i in range(nstates)]).argsort()[::-1]
    else:
        idx = np.array([obs[np.where(dtraj == i)[0]].mean() for i in range(nstates)]).argsort()
    #make a  mapping of old indices to new indices
    mapping = np.zeros(nstates)
    mapping[idx] = np.arange(nstates)
    #map the states
    new_dtraj = mapping[dtraj]
    return new_dtraj, idx

def reindex_matrix(mat:np.ndarray, reindex:np.ndarray):
    """reindex matrix based on indices in idx"""
    if len(mat.shape) == 2:
        mat = mat[reindex,:]; mat = mat[:, reindex]
    if len(mat.shape) == 3:
        mat = mat[:,reindex,:]; mat = mat[:,:, reindex]
    return mat


def sorted_eig(x, sym = False, real = True, return_check = False):
    if sym:
        lam,v = np.linalg.eigh(x)
    else:
        lam,v = np.linalg.eig(x)
    
    check_real = {}
    for i,name in zip([lam,v],"eigen_vals,eigen_vecs".split(",")):
        check = np.iscomplex(i).any()
        check_real[name] = check
        
    if real:
        lam,v = [i.real for i in [lam,v]]    

    idx = abs(lam).argsort()[::-1]
    lam = lam[idx];v = v[:,idx]
    if return_check:
        return lam, v, check_real
    else:
        return lam, v

# def ckpredict(tmat,nlags):
#     nstates = tmat.shape[0]
#     #decompose the transtition matrix into eigen values so we can propagate it by simply
#     ##raising the eigen values to power
#     lam, v = sorted_eig(tmat) #keep it all row stochastic, eigen vals will be the same either way
#     lam = np.diag(lam)
#     v_inv = np.linalg.inv(v)
#     predictions = [np.eye(nstates),tmat]
#     for i in range (2,nlags+1):
#         predictions.append(v@(lam**i)@v_inv)
#     return np.stack(predictions)

def cktest(mats:list):
    """relax each estimated matrix by one time step
    See https://deeptime-ml.github.io/trunk/api/generated/deeptime.util.validation.ck_test.html"""
    nstates = mats[0].shape[0]
    nlags = len(mats)
    predict = np.stack([np.linalg.matrix_power(mats[0],i) for i in range(nlags+2)])
    estimate = np.concatenate([predict[:3],np.stack([np.linalg.matrix_power(i,2) for i in mats[1:]])])
    return predict, estimate

def get_its(mats, tau:int):
    n = len(mats)
    est_lams = np.stack([sorted_eig(mat)[0] for mat in mats],axis = 1)[1:]
    if (est_lams<0).any():
        est_lams = abs(est_lams)
    predict = np.stack([-(tau*i)/np.log(est_lams[:,0]**i) for i in range(1,n+1)], axis = 1)
    estimate = np.stack([-(tau*i)/np.log(est_lams[:,i-1]) for i in range(1,n+1)], axis = 1)
    return predict, estimate



def plot_its(estimate:np.ndarray, estimate_error = None, lag:int = 1, dt:float=.2,unit = "ns",
             cmap = plt.cm.jet, fig_width = 10, fig_length = 6, title:str = "Implied Timescales"):
    """estimate: eigen vals estimated at integrer multiples of the lag time
    predict: eigen vals of the initial lagtime propogated via eponentiation"""
    nprocs,nsteps = estimate.shape
    
    cs = [cmap(i) for i in range(cmap.N)]
    color_list = [cs[int(i)] for i in np.linspace(10,len(cs)-20,nprocs)]
    color_list = color_list[::-1]
    fig,ax = plt.subplots(1,1, figsize = (fig_width,fig_length))
    lag_dt = np.arange(1,nsteps+1)*lag*dt
    #each iteration plots a single process at all lag times
    for est_proc, color in zip([estimate[i] for i in range(estimate.shape[0])], color_list):
        ax.plot(lag_dt, est_proc, label = "Estimate", color = color)
        ax.scatter(lag_dt, est_proc, color = color)
    if not estimate_error is None:
        for est_error, color in zip([estimate_error[:,i] for i in range(estimate_error.shape[1])], color_list):
            ax.fill_between(lag_dt, est_error[0],est_error[1], label = "Estimate", color = color, alpha = .2)
    
    ax.plot(lag_dt,lag_dt, color = "black")
    ax.fill_between(lag_dt,lag_dt, color = "gray", alpha = .5)
    ax.set_yscale("log")
    ax.set_ylabel(f"ITS ({unit})",size = 25)
    ax.set_xlabel(rf"Lag time, $\tau$ ({unit})",size = 25)
    ax.tick_params(axis = "both", labelsize = 25)
    ax.set_title(label = title, size = 30)
    return None





def plot_cktest(predict:np.ndarray, estimate:np.ndarray,lag:int, dt:float,unit = "ns",
                predict_color = "red", estimate_color = "black", predict_errors = None, estimate_errors = None,
               fill_estimate = True): 
    
    """predict+errors should be of shape [2,predict/estimate.shape] where the 0th dim is upper and lower
    confidence intervals"""
    
    
    
    if not np.all(predict[0] == np.eye(predict.shape[1])):
        predict, estimate = [np.concatenate([np.expand_dims(np.eye(predict.shape[1],predict.shape[1]), axis = 0), i]) for i in [predict, estimate]]
        if not predict_errors is None:
            predict_errors = np.concatenate([np.expand_dims(np.stack([np.eye(predict.shape[1])]*2),axis = 1),predict_errors], axis = 1)
        if not estimate_errors is None:
            estimate_errors = np.concatenate([np.expand_dims(np.stack([np.zeros((predict.shape[1],predict.shape[1]))]*2),axis = 1),estimate_errors], axis = 1)
    
    nsteps, nstates = predict.shape[:2]
    fig,axes = plt.subplots(nstates,nstates,figsize=(15,15),sharex=True,sharey=True)
    dt_lag = np.arange(nsteps)*lag*dt
    xaxis_marker = np.linspace(0,1,nsteps)
    padding_between=0.2; padding_top=0.065
    
    
    predict_label = "Predict"
    estimate_label = "Estimate"

    for i in range(nstates):
        for j in range(nstates):
            if not predict_errors is None: 
                axes[i,j].fill_between(dt_lag, predict_errors[0][:, i, j], predict_errors[1][:, i, j],
                                color=predict_color, alpha=0.4)
                
                predict_label +="      conf. 95%"
            
            if not estimate_errors is None:
                
                if fill_estimate:
                    axes[i,j].fill_between(dt_lag[1:], estimate_errors[0][1:, i, j], estimate_errors[1][1:, i, j],
                                color=estimate_color, alpha=0.4)
                else:
                    axes[i,j].errorbar(x = dt_lag,y = estimate[:,i,j], 
                                       yerr = abs(estimate_errors[:,:, i, j]),
                                        color=estimate_color, alpha=1)
                estimate_label +="      conf. 95%"
            
            axes[i,j].plot(dt_lag, predict[:,i,j], ls = "--", color = predict_color, label = predict_label)

            axes[i,j].plot(dt_lag, estimate[:,i,j], color = estimate_color,  label=estimate_label)

            axes[i,j].set_ylim(0,1)
            axes[i,j].text(0.1, 0.55, str(i+1)+' ->'+str(j+1),
                           transform=axes[i,j].transAxes, weight='bold', size=12)
            axes[i,j].set_yticks([0,.5,1], ["0", "0.5", "1"],size=12 )
            axes[i,j].set_xticks(dt_lag[[1,-1]],dt_lag[[1,-1]],)

    for axi in axes.flat:
        axi.set_xlabel(None)
        axi.set_ylabel(None)

    fig.supxlabel(rf"Lag time, $\tau$ ({unit})",x=0.5,y=.07,size = 25)
    fig.supylabel("Probability",x=.06,y=.5,size = 25)  
    handels, labels = axes[0,0].get_legend_handles_labels()
    fig.legend(handels, labels, ncol=7,loc="upper center", frameon=False,prop={'size': 25})
    plt.subplots_adjust(top=1.0-padding_top, wspace=padding_between, hspace=padding_between)

    

def plot_mat_error(mat,emat,title,unit,cbarlabel,textcolor:str = "white",cmap = "viridis",
                   ticklabs:list = None, val_text_size:int = 40, err_text_size:int = 40,
                   clims:list=[None,None], decimals:int = 2, round_int = False ):
    
        """mat = square matrix
        emat = matrix of same dim as mat with errors of values in mat
        unit = string specifying the units"""
        
        fig,ax = plt.subplots(1,figsize=(20,20))
        s = ax.imshow(mat, cmap=cmap, vmin = clims[0], vmax = clims[1])
        if ticklabs is None:
            ticklabs = [f"{i+1}" for i in range(mat.shape[0])]
        for i in range(len(mat)):
            for j in range(len(mat)):
                if round_int:
                    val = int(mat[j,i])
                    err = int(emat[j,i])
                else:
                    val = str(round(mat[j,i],decimals))
                    err = str(round(emat[j,i],decimals))
                
                ax.text(i, j,"{}".format(val),
                        va='center', ha='center', color = textcolor,size=val_text_size,weight="bold")
                ax.text(i, j,"\n\n $\pm${}{}".format(err,unit),
                        va='center', ha='center', color = textcolor,size=err_text_size,weight="bold")   
        
        ax.set_yticks(list(range(len(mat))), ticklabs,size=35)
        ax.set_xticks(list(range(len(mat))), ticklabs,size=35)
        ax.set_ylabel(r"$State_{i}$", size=45)
        ax.set_xlabel(r"$State_{j}$", size=45)
        cb = plt.colorbar(s,ax=ax,label=cbarlabel,fraction=0.046,pad=0.04)
        cb.set_label(cbarlabel, size=40)
        cb.ax.tick_params(labelsize=30)
        ax.set_title(title, size = 45)
        return   
    
    
def mfpt_mat(tmat, dt, lag):
    nstates = tmat.shape[0]
    mfpt = np.zeros((nstates,nstates))
    for i in range(nstates):
        mfpt[:,i] = deeptime.markov.tools.analysis.mfpt(tmat, i)
    mfpt=mfpt*(dt*lag)
    return mfpt


# In[3]:



#pretrained vamp(2) neural network architecture

# 
#  # start by making a flexible time lagged dataset generator that will keep track of train and validation indices depite shuffle and changes in lag time
#  
#  # this is necessary in order to train/estimate implied timescales at differing lagtimes for validating the deep reversible MSM 

# 
# 

# In[4]:


class DataSet(Dataset):
    def __init__(self, data:np.ndarray):
        super().__init__()
        self.data = data
    def __getitem__(self, idx):
        x = torch.from_numpy(self.data[idx]).float()
        return x
    def __len__(self):
        return len(self.data)
    
    def fullset(self, numpy = False, split = False):
        if split:
            assert self.data.shape[-1] == 2, "can't split data that doesn't have size 2 in dim -1"
        if numpy:
            if split:
                return [self.data[...,0], self.data[...,1]]
            else:
                return self.data
        else:
            if split:
                return [torch.from_numpy(self.data[...,i]).float for i in range(2)]
            else:
                return torch.from_numpy(self.data).float()
            
    
class TimeSeriesData:
    def __init__(self, data:np.ndarray, train_ratio:float=0.9, dt:float=1.0, shuffle = True,
                indices:"np.ndarray or path to .npy file" = None):
        self.data_ = data
        self.n_ = len(self.data_)
        self.ratio_ = train_ratio
        self.dt_ = dt
        if indices is None:
            self.idx_ = np.arange(self.n_)
            if shuffle: self.shuffle_idx()
        else:
            if isinstance(indices,str):
                indices = np.load(indices)
            self.idx_ = indices
        self.n_train_ = None
        self.n_val_ = None
        self.lag_ = None
        
    @property
    def shape(self):
        """return the shape of the input data"""
        return self.data_.shape
    def shuffle_idx(self):  
        np.random.shuffle(self.idx_)
    def __call__(self, lag, data=None)-> "Tuple of train and validation np.ndarrays ":
        assert lag < self.n_, "Lag is larger than the dataset"
        if data is None:data = self.data_
        self.lag_ = lag
        lagged_idx = self.idx_[self.idx_<self.n_-lag]
        n = len(lagged_idx)
        self.n_train_ = int(n*self.ratio_);self.n_val_ = n - self.n_train_
        train_idx, val_idx = lagged_idx[:self.n_train_], lagged_idx[-self.n_val_:]
        train_data, val_data = [np.stack([data[i], data[i+lag]], axis = -1) ##stack on last axis
                                for i in [train_idx,val_idx]]
        return train_data, val_data
    @property
    def idx(self):
        return self.idx_
    @property
    def n_train(self):
        return self.n_train_
    @property
    def n_val(self):
        return self.n_val_
    @property
    def dt(self):
        return self.dt_
    @property
    def lag(self):
        return self.lag_
    @property
    def data(self):
        return self.data_
    @property
    def ratio(self):
        return self.ratio_
    @ratio.setter
    def ratio(self, ratio):
        self.ratio_ = ratio
    


# In[5]:


class vamp_u(nn.Module):
    """module to train the weight vector, u
    the forward method should take pre-transformed instantaneous and time lagged data"""
    def __init__(self, output_dim, weight_transformation=torch.exp, **kwargs):
        super().__init__()
        self.M = output_dim
        #make the vector u a learnable parameter 
        self.u_kernel_ = nn.Parameter(torch.ones(self.M, requires_grad=True)/self.M)
        #some function to keep weights positive
        self.acti = weight_transformation
        self.trainable_parameter_ = True
    
    @property
    def trainable_parameter(self):
        return self.trainable_parameter_
    
    @trainable_parameter.setter
    def trainable_parameter(self, x):
        """input True or False a convenience function for making the u_kernel constant"""
        assert isinstance(x,bool), "input True or False"
        self.trainable_parameter_ = x
        if x:
            if self.u_kernel_.requires_grad:
                print("u_kernel is already a trainable parameter")
                pass
            else:
                self.u_kernel_.requires_grad = x
                print("u_kernel is now a trainable parameter")
        else:
            if not self.u_kernel_.requires_grad:
                print("u_kernel already does not require grad")
                pass
            else:
                self.u_kernel_.requires_grad = x
                print("u_kernel now does not require grad")

    @property
    def u_kernel(self):
        if self.u_kernel_.device.type != "cpu":
            return self.u_kernel_.detach().cpu().numpy()
        else:
            return self.u_kernel_.detach().numpy()
    @u_kernel.setter
    def u_kernel(self,u,cuda_=True):
        assert isinstance(u,(np.ndarray,torch.Tensor)), "u_kernel input must be a numpy array or torch tensor"
        device = next(self.parameters()).device
        if isinstance(u,np.ndarray):u = torch.from_numpy(u).float().to(device)
        u = nn.Parameter(u, requires_grad=True)
        with torch.no_grad():
            self.u_kernel_.copy_(u)
    
    def forward(self,x:"array[...,2] where the last dim is t,t+tau"):
        chi_t, chi_tau = x[...,0], x[...,1]
        chi_mean = chi_tau.mean(0)
        n = chi_tau.shape[0]
        corr_tau = 1/n*torch.matmul(chi_tau.T, chi_tau) #unweighted cov mat of lagged data
        kernel_u = self.acti(self.u_kernel_) #force vector u to be positive
        u = kernel_u / (kernel_u*chi_mean).sum() #normalize u weights wrt to mean of time lagged data
        v = torch.matmul(corr_tau, u) #correlation per state of timelagged data reweighted
        mu = 1/n*torch.matmul(chi_tau, u) #inner product of weight vector and all time lagged data
        #print(mu.shape)
        Sigma = torch.matmul((chi_tau*mu[:,None]).T, chi_tau) #reweighted cov mat with transformed emerical stat dist
        gamma = chi_tau*(torch.matmul(chi_tau, u))[:,None] #mult each column of the output data matrix
        C_00 = 1/n*torch.matmul(chi_t.T, chi_t)
        C_01 = 1/n*torch.matmul(chi_t.T, gamma)
        C_11 = 1/n*torch.matmul(gamma.T, gamma)
        
        
        return [u,mu,Sigma,v,C_00, C_01, C_11]


# In[6]:


class vamp_S(nn.Module):
    def __init__(self, output_dim, weight_transformation=torch.exp, renorm = True, **kwargs):
        super().__init__()
        # renorm forces the S values to be positive
        self.M = output_dim
        self.acti = weight_transformation
        self.renorm = renorm
        self.S_kernel_ = nn.Parameter(torch.ones((self.M,self.M), requires_grad=True)*0.1)
    
    @property
    def trainable_parameter(self):
        return self.trainable_parameter_
    
    @trainable_parameter.setter
    def trainable_parameter(self, x):
        """input True or False a convenience function for making the u_kernel constant"""
        assert isinstance(x,bool), "input True or False"
        self.trainable_parameter_ = x
        if x:
            if self.S_kernel_.requires_grad:
                print("u_kernel is already a trainable parameter")
                pass
            else:
                self.S_kernel_.requires_grad = x
                print("u_kernel is now a trainable parameter")
        else:
            if not self.S_kernel_.requires_grad:
                print("u_kernel already does not require grad")
                pass
            else:
                self.S_kernel_.requires_grad = x
                print("u_kernel now does not require grad")
    @property
    def S_kernel(self):
        if self.S_kernel_.device.type != "cpu":
            return self.S_kernel_.detach().cpu().numpy()
        else:
            return self.S_kernel_.detach().numpy()
    @S_kernel.setter
    def S_kernel(self,S, cuda_=True):
        assert isinstance(S,(np.ndarray,torch.Tensor)), "u_kernel input must be a numpy array or torch tensor"
        device = next(self.parameters()).device
        if isinstance(S,np.ndarray):S = torch.from_numpy(S).float().to(device)
        S = nn.Parameter(S, requires_grad=True)
        with torch.no_grad():
            self.S_kernel_.copy_(S)
            
    def forward(self, x):
        Sigma,v,C_00, C_01, C_11 = x
        # transform the kernel weights
        kernel_w = self.acti(self.S_kernel_)
        # enforce symmetry
        W1 = kernel_w + kernel_w.T
        # normalize the weights
        norm = W1 @ v
        if self.renorm:
            # make sure that the largest value of norm is < 1
            quasi_inf_norm = lambda x: torch.max(abs(x))
            W1 = W1 / quasi_inf_norm(norm)
            norm = W1 @ v
        w2 = (1 - torch.squeeze(norm)) / torch.squeeze(v)
        S = W1 + torch.diag(w2)
        # calculate K
        VampE_matrix = (S).T @ C_00 @ S @ C_11 - 2*(S).T @ C_01
        K = S @ Sigma
        # VAMP-E matrix for the computation of the loss
        return [VampE_matrix, K, S]


# In[7]:


import operator
import torch
import warnings
from itertools import chain
from torch.nn.modules import Module
from torch.nn.parallel.scatter_gather import scatter_kwargs, gather
from torch.nn.parallel.replicate import replicate
from torch.nn.parallel.parallel_apply import parallel_apply
from torch._utils import (
    _get_all_device_indices,
    _get_available_device_type,
    _get_device_index,
    _get_devices_properties
)
def _check_balance(device_ids):
    imbalance_warn = """
    There is an imbalance between your GPUs. You may want to exclude GPU {} which
    has less than 75% of the memory or cores of GPU {}. You can do so by setting
    the device_ids argument to DataParallel, or by setting the CUDA_VISIBLE_DEVICES
    environment variable."""
    device_ids = [_get_device_index(x, True) for x in device_ids]
    dev_props = _get_devices_properties(device_ids)

    def warn_imbalance(get_prop):
        values = [get_prop(props) for props in dev_props]
        min_pos, min_val = min(enumerate(values), key=operator.itemgetter(1))
        max_pos, max_val = max(enumerate(values), key=operator.itemgetter(1))
        if min_val / max_val < 0.75:
            warnings.warn(imbalance_warn.format(device_ids[min_pos], device_ids[max_pos]))
            return True
        return False

    if warn_imbalance(lambda props: props.total_memory):
        return
    if warn_imbalance(lambda props: props.multi_processor_count):
        return


class MultiForwardDataParallel(nn.Module):
    def __init__(self, module, method, device_ids=None, output_device=None, dim=0):
        """allows one to use multiple forward methods (particularly useful when used with the 
        multiforward pytorch neural network module) with  data parallel without unwrapping the underlying
        pytorch model
        
        As far as I know, this is novel and should be used with extreme caution (monitor memory usage)"""
        super().__init__()
        torch._C._log_api_usage_once("torch.nn.parallel.DataParallel")
        device_type = _get_available_device_type()
        if device_type is None:
            self.module = module
            self.device_ids = []
            return

        if device_ids is None:
            device_ids = _get_all_device_indices()

        if output_device is None:
            output_device = device_ids[0]

        self.dim = dim
        self.module = module
        self.method = method
        self.device_ids = [_get_device_index(x, True) for x in device_ids]
        self.output_device = _get_device_index(output_device, True)
        self.src_device_obj = torch.device(device_type, self.device_ids[0])

        _check_balance(self.device_ids)

        if len(self.device_ids) == 1:
            self.module.to(self.src_device_obj)

    def forward(self, *inputs, **kwargs):
        method = self.method
        with torch.autograd.profiler.record_function("DataParallel.forward"):
            if not self.device_ids:
                return getattr(self.module, method)(*inputs, **kwargs)

            for t in chain(self.module.parameters(), self.module.buffers()):
                if t.device != self.src_device_obj:
                    raise RuntimeError("module must have its parameters and buffers "
                                       "on device {} (device_ids[0]) but found one of "
                                       "them on device: {}".format(self.src_device_obj, t.device))

            inputs, kwargs = self.scatter(inputs, kwargs, self.device_ids)
            # for forward function without any inputs, empty list and dict will be created
            # so the module can be executed on one device which is the first one in device_ids
            if not inputs and not kwargs:
                inputs = ((),)
                kwargs = ({},)

            if len(self.device_ids) == 1:
                return getattr(self.module, method)(*inputs[0], **kwargs[0])
            replicas = self.replicate(self.module, self.device_ids[:len(inputs)])
            outputs = self.parallel_apply([getattr(i, method) for i in replicas], inputs, kwargs)
            return self.gather(outputs, self.output_device)
    
    def replicate(self, module, device_ids):
        return replicate(module, device_ids, not torch.is_grad_enabled())

    def scatter(self, inputs, kwargs, device_ids):
        return scatter_kwargs(inputs, kwargs, device_ids, dim=self.dim)

    def parallel_apply(self, replicas, inputs, kwargs):
        return parallel_apply(replicas, inputs, kwargs, self.device_ids[:len(replicas)])

    def gather(self, outputs, output_device):
        return gather(outputs, output_device, dim=self.dim)
    

         
class DataParallelPassthrough(torch.nn.DataParallel):
    """naive pass through version of data parallel to access attributes of 
    original neural network module without having to unwrap
    --- does  not solve the problem of needing access to mutliple forward methods ---"""
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)     
        


# In[8]:


def add_hooks(new_forward_func):
    """A decorator for for methods inside a MultifowardTorchModule to make a
    forward act like a forward call (still calling the forwards/backwards
    hooks)"""
    def wrapper(self: MultiforwardTorchModule, *args, **kwargs):
        return self(new_forward_func, self, *args, **kwargs)
    return wrapper


class MultiforwardTorchModule(nn.Module):
    """Wraps nn.Module to work with add_forward hooks. Instead of overriding
    forward and calling this module with __call__, you can just use the
    add_hooks on methods that act like a forward"""
    def forward(self, actual_forward: Callable, *args, **kwargs):
        """Calls the value passed in from the annotation. This should not be
        overridden (unless you want to create something that happens on all
        your forwards somewhat like a forward hook.)"""
        return actual_forward(*args, **kwargs)
    


# In[9]:


class ConstrainedVAMPnet(MultiforwardTorchModule):
    def __init__(self, chi, vamp_u, vamp_S, chi_state_dict = None, n_devices = None):
        super().__init__()
        self.vlu = vamp_u
        self.vls = vamp_S
        self.chi = chi
        if not chi_state_dict is None:
            self.chi.load_state_dict(plain_state_dict(
            torch.load(chi_state_dict)))
        if n_devices is None:
            n_devices = torch.cuda.device_count()
        self.device_ids = [*range(n_devices)]
        if n_devices>1:self.parallel = True
        else:self.parallel = False
        self.n_devices = n_devices
        
    def train_u(self,x:bool):
        self.vlu.trainable_parameter = x
    def train_S(self, x:bool):
        self.vls.trainable_parameter = x
    def train_chi(self, x:bool):
        self.chi.trainable_parameter = x
    def set_u(self,x):
        self.vlu.u_kernel = x
    def set_S(self, x):
        self.vls.S_kernel = x
    
    @add_hooks
    def evaluate_u(self,x:"tensor of lagged data in last dim"):
        return self.vlu(x)
    
    @add_hooks
    def evaluate_uS(self, x:"tensor of lagged data in last dim"):
        u,mu,Sigma,v,C_00, C_01, C_11 = self.vlu(x)
        VampE_matrix, K, S = self.vls([Sigma,v,C_00, C_01, C_11])
        return [u,mu,Sigma,v,C_00,C_01, C_11, VampE_matrix, K, S]
        
    @add_hooks
    def evaluate_chi(self, x):
        if self.parallel:
            x = nn.parallel.data_parallel(self.chi, x , device_ids = self.device_ids)
        else:
            x = self.chi(x)
        return x

    @add_hooks
    def forward_uS(self, x:"tensor of lagged data in last dim"):
        x = self.vlu(x)
        x = self.vls(x[2:])[0]
        return x

    @add_hooks
    def forward_all(self,x:"tensor of lagged data in last dim"):
        """calling parallel on this"""
        if self.parallel:
            x = torch.stack([nn.parallel.data_parallel(self.chi, i , device_ids = self.device_ids)
                            for i in [x[...,0],x[...,1]]], axis = -1)
        else:
            x = torch.stack([self.chi(i) for i in [x[...,0],x[...,1]]], axis = -1)
        x = self.vlu(x)
        x = self.vls(x[2:])
        return x[0]


# In[10]:


class EarlyStopping():
    """simple early stopping callback"""
    def __init__(self,patience,improvement_threshold = None, ckpt_fxn = None):
        self.n_trials_no_improvement = 0
        self.best_score = 0
        self.patience = patience
        self.improvement_threshold = improvement_threshold
        #self.ckpt_fxn = ckpt_fxn 
    def __call__(self,score):
        if score>self.best_score:
            if self.improvement_threshold is None:
                self.n_trials_no_improvement = 0
            else:
                if score-self.best_score<self.improvement_threshold:
                    self.n_trials_no_improvement += 1
                else:
                    self.n_trials_no_improvement = 0
            self.best_score = score
        else:
            self.n_trials_no_improvement += 1
        
        if self.n_trials_no_improvement > self.patience:
            return True
        else:
            return False


# In[11]:


class KoopmanModel():
    def __init__(self,
                 data_path:str = None,
                 chi_state_dict:str = None,
                 model_state_dict:str = None,
                 all_optim_state_dict:str = None,
                 uS_optim_state_dict:str = None,
                 output_dim:int = None,
                 lag:int = None,
                 epsilon:float = None,
                 batch_size:int = None,
                 epochs:int = None,
                 lrs:"list[uS_learning_rate, all_learning_rate] or dict with same keys as shown in list" = None,
                 indices:np.ndarray = None,
                 pin_memory=False,
                 num_workers = 0,
                 dt = .2,
                 unit = "ns",
                 n_devices = None,
                 out_dir = None,
                 args_dict = None,
                ):
        
        """this class assumes that there are torch.nn.Modules named chi, vamp_u and vamp_S already defined in the namespace"""
        
        #to restore
        if not args_dict is None:
            for key,value in args_dict.items():
                setattr(self, key, value)
        #to pass arguments
        else:
            self.data_path_ = data_path
            self.lag_ = lag
            self.dt = dt
            self.unit = unit
            self.epsilon_ = epsilon
            self.batch_size = batch_size
            self.output_dim = output_dim
            self.pin_memory = pin_memory
            self.num_workers = num_workers
            self.epochs = epochs
            self.out_dir = out_dir
            self.indices = indices
            self.train_val_log = dict(zip("uS,all,S".split(","), [None]*3))
            self.chi_state_dict = chi_state_dict
            self.model_state_dict = model_state_dict
            self.uS_optim_state_dict = uS_optim_state_dict
            self.all_optim_state_dict = all_optim_state_dict
            assert isinstance(lrs,(dict,list)),"must provide learning rates (lrs) as either a list or dict"
            if isinstance(lrs,list):
                self.lrs = dict(zip("uS,all".split(","), lrs))
            else:
                self.lrs = lrs
            if out_dir is None:
                self.out_dir = os.getcwd()

        #defaults
        #self.original_lag = None
        #self.lags = None
        #self.original_epsilon = None
        #self.epsilons = None
        if n_devices is None:
            self.n_devices = min(8,torch.cuda.device_count())       
        ###instantiate models###
        self.reset_model()

    
    @classmethod
    def restore_from_ckpt(cls, ckpt_dir):
        path_list = ckpt_dir.split("/")
        args_dict = cls.reset_to_ckpt(cls, ckpt_dir = ckpt_dir, return_args_dict = True)

        if "ckpt" in path_list[-1]:
            args_dict["trial_dir"] = "/".join(path_list[:-1])
            args_dict["ckpt_dir"] = ckpt_dir
        else:
            args_dict["trial_dir"] = ckpt_dir
        return cls(args_dict = args_dict)
    
    @property
    def indices(self):
        return self.indices_
    
    @indices.setter
    def indices(self,idx):
        self.data_gen = TimeSeriesData(data = self.data, indices = idx)
        self.indices_ = self.data_gen.idx
        return None

    @property
    def original_lag(self):
        if hasattr(self,"original_lag_"):
            return self.original_lag_
        else:
            return self.lag

    @original_lag.setter
    def original_lag(self, lag):
        self.original_lag_ = lag
        return None
    
    @property
    def lag(self):
        return self.lag_
    
    @lag.setter
    def lag(self, l):
        if not hasattr(self,"lag_"):
            self.lag_ = l
            pass
        if l == self.lag_:
            pass
        else:
            print(f"changing lag to {l}")
            if not hasattr(self, "lags"):
                self.lags = [self.lag_, l]
                self.original_lag = self.lag_
            else:
                self.lags+=[l]
            self.lag_ = l
        return None
    
    def reset_lag(self):
        if not self.original_lag is None:
            self.lag_ = self.original_lag
            return None
        else:
            pass
    
    @property
    def ckpt_dir(self):
        if hasattr(self, "ckpt_dir_"):
            ckpt_dir = self.ckpt_dir_
        elif hasattr(self,"trial_dir"):
            ckpt_dir = self.trial_dir
        else:
            raise Exception("This model has no checkpoint or trial dir to reset to")
        
        return ckpt_dir
    
    @ckpt_dir.setter
    def ckpt_dir(self,ckpt):
        self.ckpt_dir_ = ckpt
        pass

    @property
    def original_epsilon(self):
        if hasattr(self,"original_epsilon_"):
            return self.original_epsilon_
        else:
            return self.epsilon 
    
    @original_epsilon.setter
    def original_epsilon(self, eps):
        self.original_epsilon_ = eps
        return None
    
    @property
    def epsilon(self):
        return self.epsilon_
    
    @epsilon.setter
    def epsilon(self, eps):
        if not hasattr(self, "epsilon_"):
            self.epsilon_ = eps
            pass
        if eps == self.epsilon_:
            pass
        else:
            print(f"changing epsilon to {eps}")
            if not hasattr(self, "epsilons"):
                self.epsilons = [self.epsilon_, eps]
                self.original_epsilon = self.epsilon_
            else:
                self.epsilons+=[eps]
            self.epsilon_ = eps
        return None
    
    def reset_epsilon(self):
        if not self.original_epsilon is None:
            self.epsilon_ = self.original_epsilon
            return None
        else:
            pass
      
    @property
    def data_path(self):
        return self.data_path_
    
    @data_path.setter
    def data_path(self,data_path):
        try:
            self.data = np.load(data_path)
            self.data_path_ = data_path
        except:
            data_path = input(f"The provided data_path : {data_path} is not accessible, please enter a local path to the data used to train the model")
            self.data = np.load(data_path)
            self.data_path_ = data_path
        return None
    
    def reset_train_val_log(self,logs:"list or single key (str)" = None, all = False ):
        assert isinstance(logs, (str,list,None)), "must input a string or a list of strings for logs"
        if all:
            for key in list(self.train_val_log.keys()):
                self.train_val_log[key] = None
            return None
        
        if isinstance(logs,str):
            self.train_val_log[logs] = None
            return None
            
        if isinstance(logs, list):
            for key in logs:
                self.train_val_log[key] = None
            return None
        
        return None
    
    def save_state(self):
        ##make a new directory to store all results
        #if we haven't saved to a trial directory yet
        if not hasattr(self, "trial_dir"):
            trial_dir = f"{self.out_dir}/drmsm_trial_1"
            trial_dir = make_trial_dir(trial_dir)
            self.trial_dir = trial_dir
        
        #everything is a check point
        ckpt_dir = make_trial_dir(self.trial_dir+"/ckpt_1")
        self.ckpt_dir = ckpt_dir
        ckpt_dir+="/"

        self.reset_lag()
        self.reset_epsilon()

        hyperparameters = {"lag":self.lag, "epsilon":self.epsilon,"chi_state_dict" : self.chi_state_dict,
                           "batch_size":self.batch_size, "num_workers" :  self.num_workers,
                           "pin_memory" :  self.pin_memory, "out_dir" : self.out_dir,
                           "epochs" : self.epochs, "dt" : self.dt, "unit": self.unit,
                           "output_dim" : self.output_dim, "lrs" : self.lrs, "trial_dir":self.trial_dir,
                           "data_path":self.data_path, "train_val_log":self.train_val_log}

        ##save optim state dicts
        for key, value in self.optims.items():
            file_path = ckpt_dir+key+"_optim_state_dict.pt"
            torch.save(value.state_dict(),file_path)
            #hyperparameters[key+"_optim_state_dict"] = file_path

        #save model state_dict
        torch.save(self.model.state_dict(), ckpt_dir+"model_state_dict.pt")
        #hyperparameters["model_state_dict"] = trial_dir+"model_state_dict.pt"

        #save time lagged dataset indices
        np.save(ckpt_dir+"indices", self.indices)
        #hyperparameters["indices"] = trial_dir+"indices.npy"

        #save hyperparameters dictionary
        save_dict(ckpt_dir+"hyperparameters.pkl", hyperparameters)

        return None

    def save_results(self):
        assert hasattr(self,"results"), "There are not results bound to the class instance to save"
        if ckpt_dir is None:
            if hasattr(self,"ckpt_dir"):
                ckpt_dir = self.ckpt_dir
            else:
                self.save_state()
                ckpt_dir = self.ckpt_dir
        save_dict(self.ckpt_dir+"/results.pkl", self.results)
        return None
   
    def reset_chi(self, chi_state_dict = None):
        if chi_state_dict is None:
            assert hasattr(self,"chi_state_dict"),"Must have (the original) chi state dict to save it"
            chi_state_dict = self.chi_state_dict
        
        ###assuming that the chi class is defined in the script###
        self.model.chi = chi.load_state_dict(plain_state_dict(torch.load(chi_state_dict)))
        self.optims["all"] = Adam(self.model.parameters(), lr = self.lrs["all"])
        self.model = self.model.cuda()
        return None
    
    def reset_to_ckpt(self,ckpt_dir = None, return_args_dict = False):
        """need to load from a checkpoint directory made from this class or a directory that has only 
        one set of files for this to work easily."""
        if ckpt_dir is None:
            assert hasattr(self,"ckpt_dir"), "Must have ckpt directory to reset to"
            ckpt_dir = self.ckpt_dir

        ckpt_dir += "/"

        files = os.listdir(ckpt_dir)

        keys = "model_state_dict,all_optim_state_dict,uS_optim_state_dict,indices,hyperparameters".split(",")

        model_state_dict,all_optim_state_dict,uS_optim_state_dict,indices,hyperparameters = [ckpt_dir+f for k in keys
                                                                                             for f in files if k in f]

        hyperparameters = load_dict(hyperparameters)
        indices = np.load(indices)

        for key,val in zip(keys[:-1],[model_state_dict,all_optim_state_dict,uS_optim_state_dict,indices]):
            hyperparameters[key] = val

        if return_args_dict:
            return hyperparameters
        else:
            self.reset_model(args_dict = hyperparameters)
            return None

    def reset_model(self, args_dict = None):
        
        if hasattr(self, "model"):
            ###clear pytorch memory more reliably in an IDE
            del self.model, self.optims
            self.clean()
        
        if not args_dict is None:
            for key,value in args_dict.items():
                setattr(self,key,value)

        if self.model_state_dict is None:
            self.model = ConstrainedVAMPnet(chi = chi(),
                                            vamp_u = vamp_u(self.output_dim),
                                            vamp_S = vamp_S(self.output_dim),
                                            chi_state_dict = self.chi_state_dict,
                                            n_devices = self.n_devices).cuda()    

        else:
            self.model = ConstrainedVAMPnet(chi = chi(),
                                            vamp_u = vamp_u(self.output_dim),
                                            vamp_S = vamp_S(self.output_dim),
                                            n_devices = self.n_devices)
            self.model.load_state_dict(torch.load(self.model_state_dict))
            self.model = self.model.cuda()

        self.optims = {"uS":Adam([*self.model.parameters()][:2], lr = self.lrs["uS"]),
              "all":Adam(self.model.parameters(), lr = self.lrs["all"])}

        if not None in (self.uS_optim_state_dict, self.all_optim_state_dict):
            for k,d in zip(self.optims.keys(), [self.uS_optim_state_dict, self.all_optim_state_dict]):
                self.optims[k].load_state_dict(torch.load(d))
                self.lrs[k] = self.optims[k].param_groups[0]["lr"]
    
        self.clean()
        
        return None

    def update_train_val_log(self, key:str, increment_log = True):
        if key is None:
            
            return None
        
        empty_log = {"epoch":[], "train_scores":[], "val_scores":[], "lag":self.lag}
        
        if self.train_val_log[key] is None:
            self.train_val_log[key] = {1:empty_log}
            self.logger =  self.train_val_log[key][1]
            return None
        n = np.array([*self.train_val_log[key].keys()]).max()
        if increment_log:
            n+=1
            self.train_val_log[key][n] = empty_log
            print(f"Updated log {key} with attempt {n}")
        
        self.logger = self.train_val_log[key][n]
        return None
    
    def clean(self):
        gc.collect()
        with torch.no_grad():
            torch.cuda.empty_cache()
        return None
    
    def inv(self,x, sqrt = False):
        lam, v = np.linalg.eigh(x)
        include = lam > self.epsilon
        lam = lam[include]
        v = v[:,include]
        if sqrt:
            return v @ np.diag(1/np.sqrt(lam)) @ np.transpose(v)
        else:
            return v @ np.diag(1/lam) @ np.transpose(v)
    
    def cov(self,x,y=None):
        n = len(x)
        if y is None:
            C = (1/n)*(x.T@x)
        else:
            C = (1/n)*(x.T@y)
        return C
    
    def compute_pi(self,x):
        eigv, eigvec = np.linalg.eig(x.T)
        ind_pi = np.argmin((eigv-1)**2)
        pi_vec = eigvec[:,ind_pi]
        pi = pi_vec / np.sum(pi_vec)
        return pi
        
    def optim_weights(self, u = True, S = False, chi_data = None,
                      disable_bar = True, return_none = False):

        if chi_data is None:
            train_data, val_data = self.predict_chi(disable_bar = disable_bar, return_lagged = True)
        else:
            train_data, val_data = self.get_data(data = chi_data, loader = False)
        
        if u or S:    
            chi_0, chi_1 = train_data.fullset(numpy=True, split = True)
            c_0inv,c_01 = self.inv(self.cov(chi_0)), self.cov(chi_0, chi_1)
            K = c_0inv@c_01
            if u:
                pi = self.compute_pi(K)
                u_optimal = c_0inv@pi
                u_kernel = np.log(np.abs(u_optimal))
                self.model.set_u(u_kernel)
            if S:
                Sigma = self.predict_uS(train_data)[2]
                Sigma_inv = self.inv(Sigma)
                S_nonrev = K@Sigma_inv
                S_rev = .25*(S_nonrev+S_nonrev.T)
                S_kernel = np.log(np.abs(S_rev))
                self.model.set_S(S_kernel)

        if return_none:     
            return None
        else:
            return [train_data, val_data]
    
    def train_uS(self, optimal_u = True, optimal_S = False, train_u = True,
                 chi_data = None, epochs = None, log = "uS",increment_log=True, 
                 leave_bar = True, desc = "Constraint layer progress", disable_bar = False,
                 check_early_stop_every_n = 1, patience = 10, improvement_threshold = None,
                 new_optim = False, new_lr = None, train_its = False):
        
        self.update_train_val_log(log, increment_log = increment_log)
        
        callback = EarlyStopping(patience = patience, improvement_threshold = improvement_threshold)

        #potentially set u or S to its optimal value and get chi train and val data
        train_data, val_data = [i.fullset() for i in self.optim_weights(u = optimal_u, S = optimal_S,
                                                                        chi_data = chi_data)]
        if train_its:
            train_data = val_data
        
        #set u constraint parameter to either train or remain fixed based on usage
        self.model.train_u(train_u)
        
        #create a new optimizer for training S at different lags or default to uS optimizer for initial training
        if new_optim:
            if new_lr is None:
                new_lr = self.lrs["uS"]
            optimizer = Adam([*self.model.parameters()][:2], lr = new_lr)
        else:
            optimizer = self.optims["uS"]

        if epochs is None:
            epochs=self.epochs
        for epoch in tqdm(range(epochs), desc = desc,
         leave = leave_bar, colour = "black",
         bar_format = ("{desc} - epoch:{n_fmt}/{total_fmt} {bar}"
                       "iter duration:{rate_inv_fmt}, eta:{remaining_s:.3f} sec"),
                         disable = disable_bar):
            self.model.train()
            optimizer.zero_grad()
            x = self.model.forward_uS(train_data.cuda())
            loss = torch.trace(x)
            loss.backward()
            optimizer.step()
            
            self.logger["train_scores"].append(abs(loss.detach().cpu().item()))
            if len(self.logger["epoch"]) == 0:self.logger["epoch"].append(epoch)
            else: self.logger["epoch"].append(self.logger["epoch"][-1]+1)
                
            self.model.eval()
            with torch.no_grad():
                x = self.model.forward_uS(val_data.cuda())
                loss = torch.trace(x)
                self.logger["val_scores"].append(abs(loss.detach().cpu().item()))
            if (epoch+1)%check_early_stop_every_n == 0:
                if callback(abs(loss.detach().cpu().item())):
                    self.clean()
                    print(f"uS training was stopped at epoch {epoch}")
                    break
            
        self.clean()
        return
    
    def train_all(self,epochs = None, accumulate_grad_batches = 1,
                  log = "all", increment_log=True, leave_bar = True, desc = "Full model progress",
                  disable_bar = False, check_early_stop_every_n = 1, patience = 10,
                  improvement_threshold = 1e-4):
        
        self.update_train_val_log(log, increment_log = increment_log) #prepare logger (self.logger is changed)
        train_loader, val_loader = self.get_data() #train,val inputs for chi net
        self.model.train_u(True) #in this case we always train constraint parameter u
        callback = EarlyStopping(patience = patience, improvement_threshold = improvement_threshold)
        
        if epochs is None:
            epochs = self.epochs
        for epoch in tqdm(range(epochs), desc = desc,
         leave = leave_bar, colour = "black",
                        bar_format = ("{desc} - epoch:{n_fmt}/{total_fmt} {bar}" 
                       "iter duration:{rate_inv_fmt}, eta:{remaining_s:.3f} sec"),
                         disable = disable_bar):
            
            train_loss, train_steps, val_loss = [0]*3
            self.optims["all"].zero_grad()
            
            self.model.train()
            for i,x in enumerate(train_loader):
                x = self.model.forward_all(x.cuda())
                loss = torch.trace(x)
                loss /= accumulate_grad_batches
                loss.backward()
                #we have gradient accumulation available but is essentially off by default
                if (i+1)%accumulate_grad_batches == 0:
                    self.optims["all"].step()
                    self.optims["all"].zero_grad()
                    train_loss += loss.detach().cpu().item()
                    train_steps += 1

            self.logger["train_scores"].append(abs(train_loss/train_steps))
            if len(self.logger["epoch"]) == 0:self.logger["epoch"].append(epoch)
            else: self.logger["epoch"].append(self.logger["epoch"][-1]+1)
            
            self.model.eval()
            with torch.no_grad():
                for x in val_loader:
                    x = self.model.forward_all(x.cuda()) 
                    val_loss += torch.trace(x).detach().cpu().item()
                self.logger["val_scores"].append(abs(val_loss/len(val_loader)))
            if (epoch+1)%check_early_stop_every_n == 0:
                if callback(abs(val_loss/len(val_loader))):
                    self.clean()
                    print(f"all training was stopped at epoch : {epoch}")
                    break
        self.clean()
        return None

    def train_loop(self, epochs = 5000, disable_bar = False, disable_nested_bar = True, 
                  save_state_every_n:"interval for saving the state of the model during training (in hours)" =  5):
        #save_state_every_n where n HOURS
        self.optim_weights(u=True,S=True,return_none = True)
        callback = EarlyStopping(patience = 2)
        checkpoint = timer(save_state_every_n)
        
        for epoch in tqdm(range(epochs), desc = "Constrain/full loop progress",
             leave = True, colour = "black",
             bar_format = ("{desc} - epoch:{n_fmt}/{total_fmt} {bar}"
                           "iter duration:{rate_inv_fmt}, eta:{remaining_s:.3f} sec"),
                         disable = disable_bar):
            #assuming u and S have just been trained, train all first
            
            self.train_uS(epochs=500, increment_log = False, leave_bar = True,
                          disable_bar = disable_nested_bar)
            
            self.train_all(epochs=500, increment_log = False, leave_bar = True,
                           disable_bar = disable_nested_bar)
            
            #easiest to set a large number of epochs and let early stopping kill the run
            if checkpoint():
                self.save_state()
            if callback(self.logger["val_scores"][-1]):
                print(f"training loop was stopped at epoch {epoch}")
                self.save_state()
                self.clean()
                break
            
        return None
    
    def train_S(self, lag, epsilon = None, chi_data = None, epochs = 50, optimal_u = True, optimal_S = True,
                disable_bar = False, patience = 10, new_optim = True, new_lr = None, save_original_result = True):
        #allow for chi data to be input to avoid recalculating, fall back to recalculating
        if chi_data is None:
            chi_data = self.predict_chi()
        if not epsilon is None:
            self.epsilon = epsilon
        #if there are not yet results, ensure to get result for inital lag 
        if not hasattr(self, "results"):
            self.reset_lag()
            self.reset_epsilon()
            self.results = {}
            if save_original_result:
                train_result, val_result = self.predict_uS(chi_data=chi_data, S = True)
                train_result, val_result = train_result[-2], val_result[-2]
                self.results[self.original_lag] = {"val_result":{}, "train_result":{}}
                self.results[self.original_lag]["train_result"]["K"] = train_result
                self.results[self.original_lag]["val_result"]["K"] = val_result
                ckpt_dir = self.ckpt_dir +"/"
                for i,j in zip("train,val".split(","), [train_result, val_result]):
                    np.save(ckpt_dir+i+"_transition_matrix",j)
                    pi = self.compute_pi(j)
                    np.save(ckpt_dir+i+"_stationary_distribution",pi)
   
        #now retrain with different lagtime
        self.model.train()
        self.lag = lag
        self.train_uS(chi_data = chi_data, optimal_u = optimal_u, optimal_S = optimal_S,
                      train_u = True,epochs = epochs, log = "S",   increment_log = True,
                      leave_bar = True, desc = f"Training S constraint for lag {lag}",
                      patience = patience,new_optim = new_optim, new_lr = new_lr)
        train_result, val_result = self.predict_uS(chi_data=chi_data, S = True)
        train_result, val_result = train_result[-2], val_result[-2]
        self.results[lag] = {"val_result":{}, "train_result":{}}
        self.results[lag]["train_result"]["K"] = train_result
        self.results[lag]["val_result"]["K"] = val_result
        
        self.reset_lag()
        self.reset_epsilon()
        return None
            
    def train_S_loop(self,lags = None, epsilon = None, chi_data = None, epochs = 50, steps = 5, optimal_u = True,
                     optimal_S = True, patience = 10, new_optim = True, new_lr = None, disable_bar = True, save_original_result = True):
    
        """make set of lagtimes to validate model, increment by integer multiples of the original lagtime 
        of the chi model"""
        
        if lags is None:
            lags = np.arange(2,steps+2)*self.original_lag
        
        if chi_data is None:
            chi_data = self.predict_chi()
        
        for lag in tqdm(lags, desc = f"S constraint training progress",leave = True, colour = "black",
        bar_format = ("{desc} - epoch:{n_fmt}/{total_fmt} {bar}"
                      "iter duration:{rate_inv_fmt}, eta:{remaining_s:.3f} sec"),disable = disable_bar):
            
            self.train_S(lag = lag,epsilon = epsilon, chi_data = chi_data, epochs = epochs, optimal_u = optimal_u, optimal_S = optimal_S,
                        disable_bar = disable_bar, patience = patience, new_optim = new_optim, new_lr = new_lr, save_original_result = save_original_result)

    def predict_chi(self, return_lagged = False, return_flat = False,
                    return_lag = None, disable_bar = False):
        dataloader = self.get_data(flat=True)
        self.model.eval()
        with torch.no_grad():
            outs = []
            for x in tqdm(dataloader, desc = "Chi prediction progress",leave = True, colour = "black",
                    bar_format = ("{desc} - epoch:{n_fmt}/{total_fmt} {bar}"
                    "iter duration:{rate_inv_fmt}, eta:{remaining_s:.3f} sec"),
                         disable = disable_bar):
                x = self.model.evaluate_chi(x.cuda())
                outs.append(x.cpu().numpy())
            outs = np.concatenate(outs)

        self.clean()
        if return_flat and return_lagged:
            return outs, self.get_data(data = outs, loader = False, lag = return_lag)
        
        elif return_lagged:
            return self.get_data(data = outs, loader = False, lag = return_lag)

        else:
            return outs
        
    def save_chi_data(self,chi_data = None, ckpt_dir = None, reindex:np.ndarray = None):
        if ckpt_dir is None:
            if hasattr(self,"ckpt_dir"):
                ckpt_dir = self.ckpt_dir
            else:
                self.save_state()
                ckpt_dir = self.ckpt_dir
                
        ckpt_dir += "/"
        if chi_data is None:
            chi_data = self.predict_chi()
        if not reindex is None:
            chi_data = chi_data[:,reindex]
        np.save(ckpt_dir+"chi_data", chi_data)
        return 
           
    def ensemble_average(self, observable:np.ndarray,
                         standard_deviation = False, distribution = False,
                         nbins = 100, reindex:np.ndarray = None,
                         chi_data = None, obs_name:str = None,
                         save = False, ckpt_dir = None):
        outs = {}
        #check if there's data
        if chi_data is None:
            chi_data = self.predict_chi()
        #check if we should reindex
        if not reindex is None:
            chi_data = chi_data[:,reindex] 
        
        #check if the data is multivariate
        if len(observable.squeeze().shape) != 1:
            assert observable.shape[0]>observable.shape[1], "must input a Nsamples by Mfeatures matrix"
            assert not distribution, "computing distributions for multivariate data might blow up ram"
        else:
            observable = observable.flatten()
        
        outs["ave"] = expectation(observable,chi_data)
        
        if standard_deviation:
            outs["sd"] = expectation_sd(observable,chi_data,outs["ave"])
        
        if distribution:
            outs["dist"] = {}
            _,outs["dist"]["bin_centers"] = pmf1d(x=observable, nbins = nbins)
            outs["dist"]["probs"] = np.stack([pmf1d(x=observable, nbins = nbins, weights = i/i.sum(),return_bin_centers = False) for i in chi_data.T])
            
        #check if we want to save    
        if save:
            #check if there's a checkpoint to save to
            if ckpt_dir is None:
                if hasattr(self,"ckpt_dir"):
                    ckpt_dir = self.ckpt_dir
                else:
                    self.save_state()
                    ckpt_dir = self.ckpt_dir
            ckpt_dir+="/"
            #make sure the computed stats have a base name to save with
            assert not obs_name is None, "Must provide a string identifier (obs name) to save ensemble average data"
            #always save expectation value
            for key,value in out.items():
                if isinstance(value,np.ndarray):
                    np.save(ckpt_dir+f"{obs_name}_ensemble_{key}", value)
                else:
                    save_dict(ckpt_dir+f"{obs_name}_ensemble_{key}.pkl", value)
                    
                
        return outs
    
    def chi_weighted_hist2d(self,obs1:np.ndarray, obs2:np.ndarray,
                            obs_name1:str = None, obs_name2:str = None,
                            reindex:np.ndarray = None,chi_data = None, nbins = 30,
                            save = False, ckpt_dir = None):
        outs = {}
        
        if chi_data is None:
            chi_data = self.predict_chi()
        
        if not reindex is None:
            chi_data = chi_data[:,reindex] 
        
        assert len(obs1.squeeze().shape) == 1 and len(obs2.squeeze().shape) == 1, "both observables must be 1 dimensional"
        
        #flatten inputs to avoid menial errors
        obs1,obs2 = [i.flatten() for i in [obs1,obs2]]  
        
        #get bin centers once
        outs["y_centers"], outs["x_centers"] = [i[1:] + np.diff(i)/2 for i in np.histogram2d(x = obs2, y = obs1, bins = nbins)[1:]]
        
        #get distributions for all states
        outs["probs"] = np.stack([np.flipud(np.histogram2d(x = obs2, y = obs1, bins = nbins, weights = i/i.sum(), normed = True)[0]) for i in chi_data.T])
        
        if save:
            if ckpt_dir is None:
                if hasattr(self,"ckpt_dir"):
                    ckpt_dir = self.ckpt_dir
                else:
                    self.save_state()
                    ckpt_dir = self.ckpt_dir
                    
            ckpt_dir+="/"
            save_dict(ckpt_dir+f"{obs_name1}.{obs_name2}_ensemble_dist2d.pkl",outs)
        
        return outs
                            
    def predict_uS(self,x = None, chi_data = None, S = True, return_chi = False, lag = None):
        """all data must be in one batch (dataloader is trivial)
        if S == False (default)
        returns u,mu,Sigma,v,C_00, C_01, C_11
        if S == True
        returns u,mu,Sigma,v,C_00,C_01, C_11, VampE_matrix, K, S"""
        self.model.eval()
        outs = []
        with torch.no_grad():
            if not chi_data is None:
                """Assuming flat and ordered chi-transformed data, we make train and val data
                given a lag time in order to minimize memory usage/the need to input
                chi data in train/val format"""
                train_data, val_data = [i.fullset() for i in 
                                        self.get_data(data = chi_data, loader = False, lag = lag)]
            else:
                if not x is None:
                    if isinstance(x,(tuple,list)):
                        """ if a tuple is passed, we interpret as (train, val) and then determine the format
                        this block seeks to synch everything to torch tensors so that other blocks
                        are not replicated with slight deviations"""
                        if isinstance(x[0],np.ndarray):
                            train_data, val_data = [torch.from_numpy(i).float() for i in x]
                        if isinstance(x[0],DataSet):
                            train_data, val_data = [i.fullset() for i in x]
                    else:
                        """if we get a single data structure as input, we evaluate it after checking and
                        setting the format and assume it is just train or val data"""
                        if isinstance(x,np.ndarray):
                            x = torch.from_numpy(x).float()
                        if isinstance(x,DataSet):
                            x = x.fullset()
                        if S:
                            out = self.model.evaluate_uS(x.cuda())
                        else:
                            out = self.model.evaluate_u(x.cuda())
                        self.clean()
                        return [i.cpu().numpy() for i in out]
                else:
                    if return_chi:
                        chi_data, train_data, val_data = self.predict_chi(return_flat = True, return_lagged = True, 
                                                                         return_lag = lag)
                        outs.append(chi)
                    else:
                        train_data, val_data = self.predict_chi(return_lagged = True, return_lag = lag)
                    
                    train_data, val_data = [i.fullset() for i in [train_data, val_data]]
                
            
            if S:
                for i in [train_data, val_data]:
                    out = self.model.evaluate_uS(i.cuda()) #generates a list of tensors
                    out = [i.cpu().numpy() for i in out] #convert to numpy and put on cpu
                    outs.append(out) #append to full output   
            else:
                for i in [train_data, val_data]:
                    out = self.model.evaluate_u(i.cuda()) #generates a list of tensors
                    out = [i.cpu().numpy() for i in out] #convert to numpy put on cpu
                    outs.append(out) #append to full output
            
        self.clean()
        return outs  
                        
    def get_data(self, data=None, lag=None, batch_size=None,
                one_batch_val = False, flat = False, loader = True):
        """Will return either a DataSet or DataLoader object"""
        if flat:
            "for chi evaluation"
            dataset = DataSet(self.data)
            dataloader = torch.utils.data.DataLoader(dataset, batch_size = self.batch_size,
            pin_memory=self.pin_memory,num_workers=self.num_workers,shuffle=False,drop_last=False)
            return dataloader
        
        if lag is None: 
            lag = self.lag
        
        if data is None:
            train_dataset,val_dataset = [DataSet(i) for i in self.data_gen(lag = lag)]
        else:
            """for transformed data"""
            if isinstance(data, (tuple, list)):
                train_dataset, val_dataset = [DataSet(i) for i in data]  
            else:
                train_dataset, val_dataset = [DataSet(i) for i in self.data_gen(data = data, lag = lag)]
        
        if not loader:
            return train_dataset, val_dataset
        
        if batch_size is None:
            if one_batch_val:
                val_batch_size = len(val_dataset)
                train_batch_size = self.batch_size
            else:
                train_batch_size, val_batch_size = 2*[self.batch_size]
        
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = train_batch_size,
            pin_memory=self.pin_memory,num_workers=self.num_workers,shuffle=True,drop_last=False)

        val_loader = torch.utils.data.DataLoader(val_dataset,batch_size = val_batch_size,
            pin_memory=self.pin_memory,num_workers=self.num_workers,shuffle=False,drop_last=False)
        
        return train_loader,val_loader
        
    def plot_train_val(self, keys:str="uS,all", trial:int = 1, title:str = "VAMP-E Score"):
        keys = keys.replace("","").split(",")
        scale = len(keys)
        fig,ax = plt.subplots(2,1*scale, figsize = (12, 10))
        if scale == 1:
            ax = np.expand_dims(ax,-1)
        for i,key in enumerate(keys):
            t = self.train_val_log[key][trial]["train_scores"]
            v = self.train_val_log[key][trial]["val_scores"]
            e = self.train_val_log[key][trial]["epoch"]
            lag = self.train_val_log[key][trial]["lag"]

            #raw scores
            ax[0, i].plot(e,t, label = "Train", color = "gray")
            ax[0, i].plot(e,v, label = "Validation", color = "red")
            ax[0,i].set_title(f"{title} : {key} (trial : {trial}, lag : {lag})")
            ax[0,i].set_ylabel("VAMP-E Score")
            ax[0,i].set_xlabel("Epoch")
            ax[0,i].set_yscale("log")
            ax[0,i].legend()

            #derivative
            ax[1,i].plot(e[1:],np.diff(t), label = "Train", color = "gray")
            ax[1,i].plot(e[1:],np.diff(v), label = "Validation", color = "red")
            ax[1,i].set_title(f"{title} : {key} - Derivative (trial : {trial}, lag : {lag})")     
            ax[1,i].set_ylabel("Derivative VAMP-E Score")
            ax[1,i].set_xlabel("Epoch")
            ax[1,i].legend()
            ax[1,i].axhline(y=0, color='black', linestyle='-')

        fig.suptitle("Training Results", size = 25)
        return None
    
    def validation_tests(self,reindex = None, ck_test = True, its = False,
                         save_data = False, save_plot = False, plot = True, tau = None):
        
        assert hasattr(self, "results"), "Must have results to make CK-test"

        if ckpt_dir is None:
            if hasattr(self,"ckpt_dir"):
                ckpt_dir = self.ckpt_dir
            else:
                self.save_state()
                ckpt_dir = self.ckpt_dir

        if tau is None:
            tau = self.original_lag
            
        
        mats = []
        if reindex is None:
            reindex = np.arange(self.output_dim)
        for k in self.results.keys():
            mats.append(self.results[k]["train_result"]["K"])
        mats = [reindex_matrix(mat,reindex) for mat in mats]
        predict, estimate = cktest(mats)
        if ck_test:
            if save_data:
                ck_test_result = np.stack([predict[1:],estimate[1:]])
                np.save(out_dir+"/ck_test", ck_test_result)
            if plot:
                plot_cktest(predict, estimate, lag = tau, dt = self.dt)
                if save_plot:
                    plt.savefig(out_dir+"/ck_test.png")

        if its:
            predict_its,estimate_its = get_its(estimate[1:],tau)
            if save_data:
                its_test_result = np.stack([predict_its,estimate_its])
                np.save(out_dir+"/its_test", its_test_result)
            if plot:
                plot_its(estimate = estimate_its,lag = tau, dt = self.dt)
                if save_plot:
                    plt.savefig(out_dir+"/its_test.png")
        pass


#  # making a class instance of KoopmanModel and training the constraints, u and S

# In[ ]:


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description = "Train Constrained VAMPnet")
    
    parser.add_argument("--data_path", "-data", required = True, type = str,
                        help = "input data for neural network")
    
    parser.add_argument("--chi_state_dict", required = True, type = str,
                       help = "File path to pytorch module state dict")
    
    parser.add_argument("--n_epochs", "-n", required = True, type = int,
                        help = "Maximum Number of Epochs")
    
    parser.add_argument("--lagtime", "-lag", required = True, type = int,
                        help = "VAMPnet lag time")
    
    parser.add_argument("--epsilon", "-eps", required=True, type = float,
                        help = "epsilon parameter for VAMPnet loss function \
                        (minimum magnitude of eigen values to keep in inversions)")
    
    parser.add_argument("--out_dir", required = False, type = str, default = False,
                       help = "Directory to save trial and ")
    
    parser.add_argument("--batch_size", "-batch", required = True, type = int,
                        help = "training batch size")
    
    parser.add_argument("--stride", "-s", default = 1, type = int,
                        help = "Stride for input data of neural network")
    
    parser.add_argument("--constraint_lr", default = 0.01,
                        type = float, help = "Learning rate of optimizer")
        
    parser.add_argument("--network_lr", default = 1e-7,
                        type = float, help = "Learning rate of constraints optimizer")
    
    parser.add_argument("--output_dim", required = True, type = int, 
                       help = "The number of output states for the VAMPnet")
    
    parser.add_argument("--net_script", required = True, type = str,
                       help = "python script containing a neural network module named 'chi'")
    
    parser.add_argument("--pin_memory", required = False, type = bool,
                       help = "Bool: pin memory of pytorch dataloaders")
    
    parser.add_argument("--num_workers", required = False, type = str, default = 0,
                       help = "Number of workers to use in pytorch dataloaders")
    
    args = parser.parse_args()

    
    sys.path.insert(0,"/".join(args.net_script.split("/")[:-1]))
    module = importlib.import_module(args.net_script.split("/")[-1].replace(".py",""))
    g = globals()
    g["chi"] = getattr(module,"chi")
    
    koop = KoopmanModel(data_path = args.data_path,
                        chi_state_dict = args.chi_state_dict,
                        output_dim = args.output_dim,
                        lag = args.lagtime,
                        epsilon = args.epsilon,
                        batch_size = args.batch_size,
                        epochs = args.n_epochs, 
                        lrs = [args.constraint_lr, args.network_lr], #one for contraints one for the entire network
                        out_dir = args.out_dir,
                        pin_memory = args.pin_memory, 
                        num_workers = args.num_workers)
    #                     model_state_dict = model_state_dict,
    #                     uS_optim_state_dict = uS_optim_state_dict, 
    #                     all_optim_state_dict = all_optim_state_dict,
    #                     indices = indices,
    #                     ...)

    koop.train_loop() #will save it's state
