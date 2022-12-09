import pathlib
import typing
import torch
import torch.nn.functional as F
import math
import numpy as np
import pandas as pd
import hyperspy.api as hs
import pytorch_lightning as pl
import torchvision.transforms as transforms
from torch.utils.data import random_split, Dataset, DataLoader, ConcatDataset

class DPDatasetRadial1D(Dataset):
    def __init__(self, path:typing.Union[str, pathlib.Path], cube_root:bool=False):
        try:
            self.seddataset=hs.load(path)
        except:
            print("Please input a valid path for the sed dataset.")

        if cube_root:
            self.seddataset.data = self.seddataset.data**0.33
        
        self.sed_width, self.sed_height = self.seddataset.data.shape[:2]
        self.DP1d_size= self.seddataset.data.shape[-1]
        self.dp_sum_intensity = self.seddataset.data.sum(axis=2)
        self.seddataset_reshaped = self.seddataset.data.reshape(-1, self.DP1d_size)

    def __len__(self):
        return self.sed_width*self.sed_height
    
    def __getitem__(self, idx):
        dp = self.seddataset_reshaped[idx]
        dp = np.float32(dp)
        dp = dp / dp.max()
        # dp = np.pad(dp, (0, 13), 'constant', constant_values=(0, 0))
        # dp = np.pad(dp, (0, 10), 'constant', constant_values=(0, 0))
        dp = dp[:128]
        # dp = (dp-0.5) / 0.5
        return torch.tensor(dp).unsqueeze(0)
    
class DPdataset(Dataset):
    def __init__(self, path:typing.Union[str, pathlib.Path], cube_root:bool=False):
        try:
            self.seddataset=hs.load(path)
        except:
            print("Please input a valid path for the sed dataset.")
            
        assert len(self.seddataset.data.shape)==4, "The input dataset should be 4D."

        if cube_root:
            self.seddataset.data = self.seddataset.data**0.33
        
        self.sed_width, self.sed_height = self.seddataset.data.shape[:2]
        self.dp_height, self.dp_width = self.seddataset.data.shape[2:]
        self.dp_sum_intensity = self.seddataset.data.reshape(self.sed_width, self.sed_height ,-1).sum(axis=2)
        self.seddataset_reshaped = self.seddataset.data.reshape(-1, self.dp_height, self.dp_width)
        
        self.transform = transforms.Compose([transforms.ToPILImage(),
                                             transforms.Grayscale(),
                                             transforms.CenterCrop(size=192),
                                            #  transforms.Resize(size=(128,128)),
                                             transforms.ToTensor()])
                                             # transforms.Normalize((0.5), (0.5))])

    def __len__(self):
        return self.sed_width*self.sed_height
    
    def __getitem__(self, idx):
        dp = self.seddataset_reshaped[idx]
        dp = np.float32(dp)
        return self.transform(dp)

class DPDatasetRadial2D(DPdataset):
    def __init__(self, path:typing.Union[str, pathlib.Path], cube_root:bool=False):
        super().__init__(path, cube_root)

        class CustomPad:
            def __init__(self, pad:typing.Tuple = (0,24,0,13)):
                self.pad = pad
            def __call__(self, img):
                return F.pad(img, self.pad, mode='constant', value=0)

        self.transform = transforms.Compose([transforms.ToPILImage(),
                                             transforms.Grayscale(),
                                             transforms.ToTensor(),
                                             CustomPad(pad=(0,24,0,13)), # pad the image to 384*160
                                             transforms.Normalize((0.5), (0.5))])
       

class DPDatasetMultiChannel1D(DPdataset):
    def __init__(self, 
                 path:typing.Union[str, pathlib.Path], 
                 cube_root:bool=False, 
                 input_feature_num:int=128, 
                 random_permutation:bool=False
                ):
        super().__init__(path)
        self.input_feature_num = input_feature_num
        self.random_permutation = random_permutation
        self.cube_root = cube_root
    def __getitem__(self, idx):
        dp = self.seddataset_reshaped[idx][:self.input_feature_num,:] # shape = [147,360]->[128,360]
        dp = np.float32(dp)
        dp = dp / (dp.max())
        dp = np.cbrt(dp) if self.cube_root else dp
            
        transform = transforms.Resize(size=(self.input_feature_num,int(dp.shape[-1]/2))) # [128,360]->[128, 180]
        dp = transform(torch.tensor(dp).view(1,1,dp.shape[0],dp.shape[1]))
        
        dp = dp.squeeze().permute(1,0) # now shape = [180,128]
        
        
        if self.random_permutation:
            return  dp[torch.randperm(dp.shape[0])] # random permutation on axis 0, i.e., 180 [180,128]
        else:
            return dp
        
        
class DPDataModule(pl.LightningDataModule):
    def __init__(self, 
                 path: typing.Union[str, pathlib.Path], 
                 dataset: Dataset,
                 random_permutation:bool=False,
                 val_data_ratio: float = 0.1,
                 batch_size: int = 32,
                 n_cpu: int = 4,
                 cube_root:bool=True,
                ):
        super().__init__()
        self.path = path
        self.val_data_ratio = val_data_ratio
        self.random_permutation = random_permutation
        self.batch_size = batch_size
        self.n_cpu = n_cpu
        self.cube_root = cube_root
        self.dataset=dataset
        
        if self.dataset == DPDatasetMultiChannel1D:
            self.dataset_full = dataset(path=self.path, cube_root=self.cube_root, random_permutation=self.random_permutation)
        else:
            self.dataset_full = dataset(path=self.path, cube_root=self.cube_root)

    def setup(self, stage = None):
        if stage == "fit" or stage is None:
            all_size = self.dataset_full.__len__()
            val_size = int(all_size*self.val_data_ratio) if (self.val_data_ratio > 0.0) else int(all_size*0.1)
            self.dataset_train, self.dataset_val = random_split(self.dataset_full, [(all_size - val_size), val_size])
        if stage == "test":
            self.dataset_test = self.dataset_full

    def train_dataloader(self):
        if self.val_data_ratio > 0.0:
            return DataLoader(self.dataset_train,
                              batch_size=self.batch_size,
                              shuffle=True,
                              num_workers=self.n_cpu,
                              pin_memory=True,)
        else:
            return DataLoader(ConcatDataset([self.dataset_train, self.dataset_val]),
                              batch_size=self.batch_size,
                              shuffle=True,
                              num_workers=self.n_cpu,
                              pin_memory=True,)
    
    def val_dataloader(self):
        return DataLoader(self.dataset_val,
                        batch_size=64,
                        shuffle=False,
                        num_workers=self.n_cpu,
                        pin_memory=True,)

    def test_dataloader(self):
        return DataLoader(self.dataset_test,
                          batch_size=self.batch_size,
                          shuffle=False,
                          num_workers=self.n_cpu,
                          pin_memory=True,)
    
