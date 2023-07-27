#!/usr/bin/env python3
from torch.utils.data import DataLoader, Dataset, TensorDataset
import pytorch_lightning as pl
import deeptime
from deeptime.decomposition import VAMP
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import sys
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger as logger_
import matplotlib.colors as colors
from tqdm import tqdm
import importlib
def plain_statedict(d, badword):
    from collections import OrderedDict
    new = OrderedDict()
    for k,v in d.items():
        name = k.replace(badword+".", "")
        new[name] = v
    return new

def get_metrics(path):
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    event_accumulator = EventAccumulator(path)
    event_accumulator.Reload()
    steps = {x.step for x in event_accumulator.Scalars("epoch")}
    epoch = list(range(len(steps)))
    train_loss, val_loss = [-np.array([x.value for x in event_accumulator.Scalars(_key) if x.step in steps]) for _key in
                            ["train_loss","val_loss"]]
    return np.array(epoch),train_loss, val_loss


def proj2d(p,c,state_map = False, ax = None, filename=None):
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
        s = ax.scatter(p[:,0],p[:,1],c=c,s=.5,cmap="jet")
        cbar = plt.colorbar(s,ax=ax)
        cbar.set_label("Probability",size=15)
    ax.set_xlabel("Comp. 1",fontsize=15)
    ax.set_ylabel("Comp. 2",fontsize=15)
    ax.tick_params(axis="x",labelsize=10)
    ax.tick_params(axis="y",labelsize=10)
    cbar.ax.tick_params(labelsize=8)
    if not filename is None:
        plt.savefig(filename)
    #plt.clf()
    return


if __name__ == "__main__":
    
    ##parse arguments
    
    parser = argparse.ArgumentParser(description = "Restore pytorch lightning model, compute plots of train/validation data and \
                                     initial VAMP observables. This script will make a new directory inside \
                                      'dir' (argument) named 'name'.results (using the 'name' argument as prefix")
    parser.add_argument("--dir", required = True,type = str,
                        help = "Directory with saved checkpoint and log")
    
    parser.add_argument("--name", required = True, type = str,
                        help = "Root name for checkpoint file and log directory, should be the same except \
                        .ckpt (for lightning setup, will likely be version_#)")

    parser.add_argument("--data_path", required=True, type = str, 
                       help = "Path to the data to be transformed by neural network (must be a numpy array)")
    
    parser.add_argument("--batch_size", required=False, default = 16384, type = int, 
                       help = "The number of samples to evaluate in a given batch")

    parser.add_argument("--net_script", required = True, type = str,
                       help = "python script containing a neural network module named 'chi' or pass the name of the pytorch module via the net_name argument")

    parser.add_argument("--net_name", default = "chi",type = str,
                        help = "name of the pytorch module defined in net_script to use in the vampnet")

    parser.add_argument("--latent_space", default = None, type = str,
                        help = ".npy file containing latent space coordinates (2D or higher) to project state probabilities onto")

    args = parser.parse_args()

    sys.path.insert(0,"/".join(args.net_script.split("/")[:-1]))
    module = importlib.import_module(args.net_script.split("/")[-1].replace(".py",""))
    g = globals()
    g["chi"] = getattr(module,args.net_name)

    ##make a new directory to store all results
    newdir = args.dir+"/"+args.name+".results"
    if not os.path.exists(newdir):
        os.makedirs(newdir)
    newdir+="/" #to be lazy later
    
    ##prepare data
    
    data = np.load(args.data_path)
    dataset = TensorDataset(torch.from_numpy(data).float())
    loader = DataLoader(dataset, batch_size = args.batch_size)
    
    ##get checkpoint and log information
    log_str = args.dir+"/"+args.name
    ckpt_str = log_str+".ckpt"
    last_ckpt_str = args.dir+"/"+"last.ckpt"
    epoch, train_loss, val_loss = get_metrics(log_str) #get train and val data
    epoch = epoch[:len(train_loss)]
    #plot train and val data
    plt.figure()
    plt.plot(epoch, train_loss, color = "gray", label = "Train Score")
    plt.plot(epoch, val_loss, color = "red", label = "Val Score")
    plt.xlabel("Epoch"); plt.ylabel("VAMP2 Score")
    plt.legend()
    plt.savefig(newdir+"test.val.png")
    plt.clf()
    #use checkpoint to restore regular pytorch module
    for path, name in zip([ckpt_str, last_ckpt_str], ["best", "last"]):
        ckpt = torch.load(path)
        model = chi()
        model.load_state_dict(plain_statedict(ckpt["state_dict"], "model"))
        model = model.cuda()
        if torch.cuda.device_count()>1:
            model = nn.DataParallel(model, device_ids = [*range(torch.cuda.device_count())])
        probs = []
        model.eval()
        with torch.no_grad():
            for b in tqdm(loader, desc = "Predicting Dataset",leave = False):
                out = model(b[0].cuda())
                out = out.cpu().numpy()
                probs.append(out)
        probs = np.concatenate(probs)

        v = VAMP(lagtime=ckpt["hyper_parameters"]["lagtime"]).fit_from_timeseries(probs)
        projs = v.transform(probs)
        dtraj = probs.argmax(1)
        for i,j in zip("probs,projs,dtraj".split(","), [probs, projs, dtraj]):
            np.save(f"{newdir}{i}.{name}", j)

        if args.latent_space is None:
            projection = projs
        else:
            projection = np.load(args.latent_space)

        fig,axes = plt.subplots(3,4, figsize = (20,20))
        for i,ax in zip(probs.T,axes.flat):
            proj2d(projection,c=i,ax=ax)
        plt.savefig(f"{newdir}distributions.{name}.png")

        proj2d(projs, c = dtraj,  state_map = True, filename = f"{newdir}statemap.{name}.png")
        plt.clf()
