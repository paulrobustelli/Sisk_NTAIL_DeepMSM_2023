#!/usr/bin/env python3
# coding: utf-8

# In[ ]:


#here we make the number and width of layers in the linear layers used for helical inputs 
import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import deeptime
from torch.utils.data import DataLoader
from tqdm import tqdm
from deeptime.decomposition.deep import VAMPNet
from deeptime.decomposition import VAMP
from copy import deepcopy
from deeptime.util.data import timeshifted_split
import optuna
from optuna.integration import PyTorchLightningPruningCallback
import pytorch_lightning as pl
import argparse
from optuna.integration import TorchDistributedTrial
import warnings
import pandas as pd
#warnings.filterwarnings("ignore", "*does not have many workers*")
#warnings.filterwarnings("ignore", "*find_unused_parameters=True was specified in DDP constructor*")
#warnings.filterwarnings("ignore", "*ddp_spawn and num_workers=0 may*")
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.backends.cudnn.benchmark = True
else:
    device = torch.device("cpu")




from pytorch_lightning import LightningModule
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import Callback

class PyTorchLightningPruningCallback(Callback):
    """PyTorch Lightning callback to prune unpromising trials.
    See `the example <https://github.com/optuna/optuna-examples/blob/
    main/pytorch/pytorch_lightning_simple.py>`__
    if you want to add a pruning callback which observes accuracy.
    Args:
        trial:
            A :class:`~optuna.trial.Trial` corresponding to the current evaluation of the
            objective function.
        monitor:
            An evaluation metric for pruning, e.g., ``val_loss`` or
            ``val_acc``. The metrics are obtained from the returned dictionaries from e.g.
            ``pytorch_lightning.LightningModule.training_step`` or
            ``pytorch_lightning.LightningModule.validation_epoch_end`` and the names thus depend on
            how this dictionary is formatted.
    """

    def __init__(self, trial: optuna.trial.Trial, monitor: str, best_score:float) -> None:
        super().__init__()

        self._trial = trial
        self.monitor = monitor
        self.best_score = best_score
    def on_validation_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        # When the trainer calls `on_validation_end` for sanity check,
        # do not call `trial.report` to avoid calling `trial.report` multiple times
        # at epoch 0. The related page is
        # https://github.com/PyTorchLightning/pytorch-lightning/issues/1391.
        if trainer.sanity_checking:
            return

        epoch = pl_module.current_epoch

        current_score = trainer.callback_metrics.get(self.monitor)

        
        if current_score is None:
            message = (
                "The metric '{}' is not in the evaluation logs for pruning. "
                "Please make sure you set the correct metric name.".format(self.monitor)
            )
            #warnings.warn(message)
            return
        
        #if self.monitor == "train_loss":
        if torch.isnan(current_score):
            #self._trial.report(100,step=epoch)
            raise optuna.TrialPruned(f"Trial was pruned at epoch {epoch} because training loss went to nan")
        
        if abs(current_score)>(abs(self.best_score)+0.5):
            #self._trial.report(100,step=epoch)
            raise optuna.TrialPruned(f"Trial was pruned at {epoch} because the scoring function took on a value with an erroneously large magnitude")
        if self.monitor == "val_loss":
            self._trial.report(current_score, step=epoch)
            if self._trial.should_prune():
                message = "Trial was pruned at epoch {}.".format(epoch)
                raise optuna.TrialPruned(message)
        



class Net(nn.Module):
    def __init__(self, n_linears_1:int = 4, n_linears_2:int = 4,
                 n_linears_3:int = 3, n_convs:int = 4, n_states:int = 10):
        super().__init__()
        
        ###convolutional layers###
        
        kernels = [(2,2), (2,2), (3,1), (3,2)]
        conv_out_dim = 1
        conv_dims = [32, 64, 128, 264]
        conv_layers = []
        
        ###make conv lobe in a loop with zip to allow variable number of layers###
        
        for i,d,k in zip(range(n_convs), conv_dims, kernels):
            conv_layers.append(nn.Conv2d(conv_out_dim, d, k, padding = 1))
            conv_layers.append(nn.ReLU(inplace=True))
            conv_layers.append(nn.MaxPool2d(kernel_size = k))
            conv_out_dim = d
        self.conv_lobe = nn.Sequential(*conv_layers)
        
        
        ###determine flattened output dimension of convolutional layers###
        
        self.to_linear=None
        y = torch.randn(49,21).view(-1,1,49,21)
        self.convs(y)
        self.fc1 = nn.Linear(self.to_linear,self.to_linear)

        ###first vectorized input###
        
        linear_dims = [30,50,100,160]
        linear_1_layers = []
        linear_1_out_dim = 16
        for i,d in zip(range(n_linears_1), linear_dims):
            linear_1_layers.append(nn.Linear(linear_1_out_dim, d))
            linear_1_layers.append(nn.ReLU(inplace = True))
            linear_1_out_dim = d
        self.linear_1_lobe = nn.Sequential(*linear_1_layers)
        
        ###second vectorized input###
        
        linear_2_layers = []
        linear_2_out_dim = 19
        for i,d in zip(range(n_linears_2), linear_dims):
            linear_2_layers.append(nn.Linear(linear_2_out_dim, d))
            linear_2_layers.append(nn.ReLU(inplace = True))
            linear_2_out_dim = d
        self.linear_2_lobe = nn.Sequential(*linear_2_layers)
        
        ###aggregated linear lobe###
        
        linear_3_layers = []
        linear_3_out_dim = linear_1_out_dim + linear_2_out_dim + self.to_linear
        linear_3_dims = [1072,1000, 512]
        for i,d in zip(range(n_linears_3), linear_3_dims):
            linear_3_layers.append(nn.Linear(linear_3_out_dim, d))
            linear_3_layers.append(nn.ReLU(inplace = True))
            linear_3_out_dim = d
        
        linear_3_layers.append(nn.Linear(linear_3_out_dim, n_states))
        linear_3_layers.append(nn.Softmax(1))
        self.linear_3_lobe = nn.Sequential(*linear_3_layers)
    
    ###run convs and flatten###

    def convs(self,x):
        x = self.conv_lobe(x)
        if self.to_linear is None:
            self.to_linear = x[0].shape[0]*x[0].shape[1]*x[0].shape[2]
        return x.view(-1,self.to_linear)
    
    ###forward method###
    def forward(self,x):
        image = x[:,:,:-2,:]
        image = self.convs(image)
        image = F.relu(self.fc1(image))
        vector = x[:,:,-2:-1,:-5].reshape(-1,16)
        vector = self.linear_1_lobe(vector)
        vector2 = x[:,:,-1:,:-2].reshape(-1,19)
        vector2 = self.linear_2_lobe(vector2)
        combined = torch.cat((image, vector,vector2), 1)
        combined = self.linear_3_lobe(combined)
        return combined

class LitNet(pl.LightningModule):
    def __init__(self,epsilon,lr, n_linears_1, n_linears_2, n_linears_3, n_convs, n_states):
        super().__init__()
        self.model = Net(n_linears_1, n_linears_2, n_linears_3, n_convs, n_states)
        self.epsilon = epsilon
        self.lr = lr
    def forward(self, x):
        self.model(x)
    def training_step(self,batch, batch_idx):
        b0, bt = batch; b0,bt = self.model(b0), self.model(bt)
        loss = deeptime.decomposition.deep.vampnet_loss(b0,bt,epsilon=self.epsilon,mode = "regularize")
        self.log("train_loss", loss, sync_dist=True, on_epoch=True,on_step=False)
        return loss
    def validation_step(self, batch,batch_idx):
        b0, bt = batch; b0,bt = self.model(b0), self.model(bt)
        loss = deeptime.decomposition.deep.vampnet_loss(b0,bt,epsilon=self.epsilon,mode = "regularize")
        self.log("val_loss", loss, sync_dist=True, on_epoch=True,on_step=False)
        #self.log("hp_metric", loss, on_step=False, on_epoch=True, sync_dist=True)
        return 
    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(),self.lr)
    
class DataModule(pl.LightningDataModule):
    def __init__(self, data_path, batch_size, lagtime, stride):
        super().__init__()
        self.data_path = data_path
        self.batch_size = batch_size
        self.lagtime = lagtime
        self.stride = stride
    def setup(self, stage = None):
        self.data = np.load(self.data_path)[::self.stride];self.data = torch.from_numpy(self.data).float()
        self.dataset = deeptime.util.data.TrajectoryDataset(lagtime=self.lagtime,trajectory=self.data)
        self.n_val = int(len(self.dataset)*.3)
        self.train_dataset,self.val_dataset = torch.utils.data.random_split(self.dataset,
                                                                            [len(self.dataset)-self.n_val,self.n_val])
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle = True)
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle = False)
    
def objective(trial, data_path, lagtime, epsilon, lr,batch_size, stride, max_epochs,accumulate_grad_batches,
              best_score,linear_layers_1:bool, linear_layers_2:bool, linear_layers_3:bool, conv_layers:bool):
    #trial = TorchDistributedTrial(trial)
    hyperparameters = {}
    if linear_layers_1:
        n_linears_1 = trial.suggest_int("n_layers_linear_1", 1,4)
        hyperparameters["n_layers_linear_1"] = n_linears_1
    else:
        n_linears_1 = 4
    if linear_layers_2:
        n_linears_2 = trial.suggest_int("n_layers_linear_2", 1,4)
        hyperparameters["n_layers_linear_2"] = n_linears_2
    else:
        n_linears_2 = 4
    if linear_layers_3:
        n_linears_3 = trial.suggest_int("n_layers_linear_3", 1,3)
        hyperparameters["n_layers_linear_3"] = n_linears_3
    else:
        n_linears_3 = 3
    if conv_layers:
        n_convs = trial.suggest_int("n_layers_conv", 1,4)
        hyperparameters["n_layers_conv"] = n_convs
    else:
        n_convs = 4
    
    val_callback = PyTorchLightningPruningCallback(trial, monitor="val_loss",best_score=best_score)
    train_callback = PyTorchLightningPruningCallback(trial, monitor="train_loss", best_score =best_score)
    
    model = LitNet(epsilon=epsilon,lr=lr, n_linears_1=n_linears_1, n_linears_2 = n_linears_2, n_linears_3=n_linears_3, n_convs=n_convs, n_states = best_score)
    datamodule = DataModule(data_path=data_path, lagtime=lagtime, batch_size=batch_size, stride = stride)
    trainer = pl.Trainer(
        logger = True, 
        max_epochs = max_epochs,
        enable_checkpointing = False,
        accelerator="gpu",
        devices = torch.cuda.device_count(),
        strategy = "ddp_spawn",
        callbacks = [val_callback, train_callback],
        accumulate_grad_batches = accumulate_grad_batches)
    
    ##call backs implicit in the optuna optimization algo not compatible with current pytorch lightning (try 1.5<version<1.6) 
    trainer.logger.log_hyperparams(hyperparameters)
    trainer.fit(model, datamodule = datamodule)
    #val_callback.check_pruned()
    #train_callback.check_pruned()
    return trainer.callback_metrics["val_loss"].item()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "Pytorch Lightning ddp optuna hyper parameter optimization script")
    parser.add_argument("--n_trials", "-n", required = True,type = int, 
                        help = "Number of trials to be used in parameter optimization")
    parser.add_argument("--data_path", "-data", required = True, type = str, help = "input data for neural network")
    parser.add_argument("--lagtime", "-lag", required = True, type = int, help = "VAMPnet lag time")
    parser.add_argument("--epsilon", "-eps", required=True, type = float, help = "epsilon (regularization) parameter for VAMPnet loss function (add small about to diagonal of koopman matrix)")
    parser.add_argument("--batch_size", "-batch", required = True, type = int, help = "training batch size")
    parser.add_argument("--stride", "-s", default = 1, type = int, help = "Stride for input data of neural network")
    parser.add_argument("--max_epochs", "-epochs", default = 100, type = int, help = "Maximum number of epochs to train trial neural network given a set of test parameters (if trial survives pruning)")
    parser.add_argument("--learning_rate", "-lr", default = 5e-6, type = float, help = "Learning rate of optimizer")    
    parser.add_argument("--best_score", "-max", default = 9, type = int,
    help = """The magnitude of the maximum possible value of the scoring function (this option only applies to scorng functions that range from [0,abs(pre-defined maximum)].
            is here to prune trials that generate erroneously high scores due to numerical error and non-convergent numerical solutions. In the case of a VAMPnet, this value will also be used as the number of 
            output states""")
    parser.add_argument("--linear_layers_1", "-l1", default = False, type = bool, help = "bool indicating if the number of linear layers in linear lobe 1 should be optimized")
    parser.add_argument("--linear_layers_2", "-l2", default = False, type = bool, help = "bool indicating if the number of linear layers in linear lobe 2 should be optimized")
    parser.add_argument("--linear_layers_3", "-l3", default = False, type = bool, help = "bool indicating if the number of linear layers in linear lobe 3 should be optimized")
    parser.add_argument("--conv_layers", "-conv", default = False, type = bool, help = "bool indicating if the number of convolutional layers in the convolutional lobe should be optimized")
    parser.add_argument("--accumulate_grad_batches", type = int, default = 1, help = "number of batches to accumulate gradients over")

    args = parser.parse_args()
    
    pruner = optuna.pruners.MedianPruner(n_startup_trials = 5, n_warmup_steps = 35, n_min_trials = 10)
    study_name = ":".join([i for i,j in zip("linear1,linear2,linear3,conv".split(","), [args.linear_layers_1, args.linear_layers_2, args.linear_layers_3, args.conv_layers]) if j])
    study = optuna.create_study(
        study_name = study_name, 
        direction = "minimize",
        pruner=pruner, 
        load_if_exists =False)
    
    func = lambda trial: objective(trial, args.data_path, args.lagtime, args.epsilon,args.learning_rate, args.batch_size,args.stride, args.max_epochs,args.accumulate_grad_batches,
                                    args.best_score, args.linear_layers_1, args.linear_layers_2, args.linear_layers_3, args.conv_layers)
    
    study.optimize(func, n_trials=args.n_trials, show_progress_bar = True,n_jobs = 1)
    print(f"Numer of finished trials: {len(study.trials)}")
    
    print("Best trial:")
    
    trial = study.best_trial
    print(f"  Value: {trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"   {key}:{value}")

    df = study.trials_dataframe()
    df.to_csv(os.getcwd()+f"/study.{study_name}.results.csv", index=False)
