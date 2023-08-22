#!/usr/bin/env python3
from torch.utils.data import DataLoader, Dataset, TensorDataset
import pytorch_lightning as pl
import deeptime
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import argparse
import os
import sys
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger as logger_
from pytorch_lightning.strategies.ddp import DDPStrategy
from collections import OrderedDict
import importlib


def source_module_attr(module_file: str, attr_name: str, local_attr_name: str = None):
    """to add a module from a user defined python script into the local name space"""

    #
    if local_attr_name is None:
        local_attr_name = attr_name

    if len(module_file.split("/")) == 1 or module_file.split("/")[-2] == ".":
        module_dir = os.getcwd()
    else:
        module_dir = "/".join(module_file.split("/")[:-1])

    sys.path.insert(0, module_dir)

    module = importlib.import_module(module_file.split("/")[-1].replace(".py", ""))

    g = globals()
    g[local_attr_name] = getattr(module, attr_name)

    pass


class TimeSeriesData:
    def __init__(self, data: np.ndarray, train_ratio: float = 0.7, dt: float = 1.0, shuffle=True,
                 indices: "np.ndarray or path to .npy file" = None):
        self.data_ = data
        self.n_ = len(self.data_)
        self.ratio_ = train_ratio
        self.dt_ = dt
        if indices is None:
            self.idx_ = np.arange(self.n_)
            if shuffle: self.shuffle_idx()
        else:
            if isinstance(indices, str):
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

    def __call__(self, lag, data=None) -> "Tuple of train and validation np.ndarrays ":
        assert lag < self.n_, "Lag is larger than the dataset"
        if data is None: data = self.data_
        self.lag_ = lag
        lagged_idx = self.idx_[self.idx_ < self.n_ - lag]
        n = len(lagged_idx)
        self.n_train_ = int(n * self.ratio_)
        self.n_val_ = n - self.n_train_
        train_idx, val_idx = lagged_idx[:self.n_train_], lagged_idx[-self.n_val_:]
        train_data, val_data = [np.stack([data[i], data[i + lag]], axis=-1)  ##stack on last axis
                                for i in [train_idx, val_idx]]
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


class DataSet(Dataset):
    def __init__(self, data: np.ndarray):
        super().__init__()
        self.data = data

    def __getitem__(self, idx):
        x = torch.from_numpy(self.data[idx]).float()
        return x

    def __len__(self):
        return len(self.data)

    def fullset(self, numpy=False, split=False):
        if split:
            assert self.data.shape[-1] == 2, "can't split data that doesn't have size 2 in dim -1"
        if numpy:
            if split:
                return [self.data[..., 0], self.data[..., 1]]
            else:
                return self.data
        else:
            if split:
                return [torch.from_numpy(self.data[..., i]).float for i in range(2)]
            else:
                return torch.from_numpy(self.data).float()


def plain_state_dict(d, badword="module"):
    new = OrderedDict()
    for k, v in d.items():
        name = k.replace(badword + ".", "")
        new[name] = v
    return new


class LitNet(pl.LightningModule):
    def __init__(self, epsilon, learning_rate, lagtime, batch_size, output_dim: int):
        super().__init__()
        self.model = chi(output_dim=output_dim)
        self.epsilon = epsilon
        self.learning_rate = learning_rate
        self.lagtime = lagtime
        self.batch_size = batch_size
        self.save_hyperparameters()

    def forward(self, x):
        self.model(x)

    def training_step(self, batch, batch_idx):
        b0, bt = batch[..., 0], batch[..., 1];
        b0, bt = self.model(b0), self.model(bt)
        loss = deeptime.decomposition.deep.vampnet_loss(b0, bt, epsilon=self.epsilon, mode="regularize")
        self.log("train_loss", loss, sync_dist=True, on_epoch=True, on_step=False)
        return loss

    def validation_step(self, batch, batch_idx):
        b0, bt = batch[..., 0], batch[..., 1];
        b0, bt = self.model(b0), self.model(bt)
        loss = deeptime.decomposition.deep.vampnet_loss(b0, bt, epsilon=self.epsilon, mode="regularize")
        self.log("val_loss", loss, sync_dist=True, on_epoch=True, on_step=False)
        # self.log("hp_metric", loss, on_step=False, on_epoch=True, sync_dist=True)
        return

    def test_step(self, batch, batch_idx):
        pass

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), self.learning_rate)


class DataModule(pl.LightningDataModule):
    def __init__(self, data_path, batch_size, lagtime, stride, indices=None):
        super().__init__()
        self.data_path = data_path
        self.batch_size = batch_size
        self.lagtime = lagtime
        self.stride = stride
        self.indices = indices

    def setup(self, stage=None):
        self.data = np.load(self.data_path)[::self.stride]
        self.data_gen = TimeSeriesData(data=self.data, indices=self.indices)
        self.train_dataset, self.val_dataset = [DataSet(i) for i in self.data_gen(lag=self.lagtime)]

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=len(self.val_dataset), shuffle=False)

    def test_dataloader(self):
        return DataLoader(TensorDataset(self.data), batch_size=self.batch_size, shuffle=False)


def main(n_epochs, data_path, lagtime, epsilon, batch_size, stride,
         learning_rate, patience, accumulate_grad_batches, filename,
         ckpt_path, training_method, output_dim, indices=None, n_devices=None):
    num_devices_available = torch.cuda.device_count()
    if n_devices is None:
        n_devices = num_devices_available
    else:
        if n_devices > num_devices_available:
            print(f"There are only {num_devices_available} devices but you have requested {n_devices}")
            print(f"Defaulting to {num_devices_available} devices")
            n_devices = num_devices_available
        elif n_devices < num_devices_available:
            print(f"You're using {n_devices} devices but there are {num_devices_available} available")
            print(f"Proceeding with {n_devices} devices")
        else:
            pass

    if training_method == "ddp":
        training_method = DDPStrategy(find_unused_parameters=False)
    if training_method == "dp":
        n_devices = min(8, n_devices)
    if n_devices == 1:
        training_method = None

    model = LitNet(epsilon=epsilon, learning_rate=learning_rate, lagtime=lagtime, batch_size=batch_size,
                   output_dim=output_dim)
    datamodule = DataModule(data_path=data_path, lagtime=lagtime, batch_size=batch_size, stride=stride, indices=indices)

    logger = logger_(save_dir="./",
                     name=filename)  # logger makes a directory, in this function name means directory name

    early_stop_ = EarlyStopping(monitor="val_loss", patience=patience, check_finite=True, mode="min")

    model_check_ = ModelCheckpoint(save_top_k=1, dirpath="./" + filename, save_last=True, monitor="val_loss",
                                   mode="min", filename=f"version_{logger.version}")

    trainer = pl.Trainer(
        logger=logger,
        accumulate_grad_batches=accumulate_grad_batches,
        max_epochs=n_epochs,
        enable_checkpointing=True,
        accelerator="gpu",
        devices=n_devices,
        check_val_every_n_epoch=1,
        callbacks=[early_stop_, model_check_],
        strategy=training_method)
    # num_processes = 1)
    trainer.fit(model, datamodule=datamodule, ckpt_path=ckpt_path)
    torch.cuda.empty_cache()
    np.save(filename + f"/version_{logger.version}.indices", datamodule.data_gen.idx)
    pass


if __name__ == '__main__':
    # get all of the arguments from the command line
    parser = argparse.ArgumentParser(description="ddp script to run a vampnet on n gpus")

    parser.add_argument("--n_epochs", "-n", required=True, type=int,
                        help="Number of trials to be used in parameter optimization")

    parser.add_argument("--data_path", "-data", required=True, type=str,
                        help="input data for neural network")

    parser.add_argument("--lagtime", "-lag", required=True, type=int,
                        help="VAMPnet lag time")

    parser.add_argument("--epsilon", "-eps", required=True, type=float,
                        help="epsilon (regularization) parameter for VAMPnet loss function \
                        (add small about to diagonal of koopman matrix)")

    parser.add_argument("--batch_size", "-batch", required=True, type=int,
                        help="training batch size")

    parser.add_argument("--stride", "-s", default=1, type=int,
                        help="Stride for input data of neural network")

    parser.add_argument("--learning_rate", "-lr", default=5e-6,
                        type=float, help="Learning rate of optimizer")

    parser.add_argument("--patience", required=False, type=int, default=100,
                        help="number of epochs to continue training after val score does not increase before early stopping")

    parser.add_argument("--accumulate_grad_batches", type=int, default=1,
                        help="number of batches to accumulate gradients over")

    parser.add_argument("--ckpt_path", default=None, type=str,
                        help="path to saved check point file for resuming training, defaults to None and is not required")

    parser.add_argument("--indices", default=None, type=str,
                        help="path to saved numpy file containing shuffled indices to be used as input to TimeSeriesData class instance")

    parser.add_argument("--net_script", required=True, type=str,
                        help="python script containing a neural network module named 'chi' or pass the name of the pytorch module via the net_name argument")

    parser.add_argument("--net_name", default="chi", type=str,
                        help="name of the pytorch module defined in net_script to use in the vampnet")

    parser.add_argument("--n_devices", type=int, default=None,
                        help="Number of (cuda enabled) devices to use")

    parser.add_argument("--distributed_training_method", type=str, default=None,
                        help="The distributed training method to implement (options: ddp (Distributed Data Parallel Data), dp (Data Parallel), None (single gpu")

    parser.add_argument("--output_dim", type=int,
                        help="The number of output states")
    # parser.add_argument("--environ_var", type=Bool, default=False,
    #                     help='Make an environmental variable (in the current shell) called ckptdir, set equal to the directory containing the state_dict of the trained neural network')

    args = parser.parse_args()

    source_module_attr(module_file=args.net_script, attr_name=args.net_name, local_attr_name="chi")

    # make a file name to save the best model
    filename = (
        f"""max_epochs:{args.n_epochs}.lagtime:{args.lagtime}.epsilon:{args.epsilon}.batch_size:{args.batch_size}."""
        f"""accum_grad_batches:{args.accumulate_grad_batches}.stride:{args.stride}.learning_rate:{args.learning_rate}.output_dim:{args.output_dim}""")
    if not os.path.exists(filename):
        os.makedirs(filename)
    # run the main function to train the model
    main(n_epochs=args.n_epochs, data_path=args.data_path, lagtime=args.lagtime, epsilon=args.epsilon,
         batch_size=args.batch_size,
         stride=args.stride, learning_rate=args.learning_rate, patience=args.patience,
         accumulate_grad_batches=args.accumulate_grad_batches,
         filename=filename, ckpt_path=args.ckpt_path, indices=args.indices, n_devices=args.n_devices,
         output_dim=args.output_dim,
         training_method=args.distributed_training_method)
