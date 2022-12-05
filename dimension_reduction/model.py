from __future__ import division
from itertools import chain

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0,0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

class VariationalAutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.apply(weights_init)


    def reparameterize(self, mu, logvar):
        std = torch.exp(logvar / 2)
        # esp = torch.randn(mu.size()).type_as(std)
        # z = mu + std * esp
        q = torch.distributions.Normal(mu, std)
        z = q.rsample()
        return z  

    def forward(self, x):
        encoder_out = self.encoder(x)
        
        mu = self.mu(encoder_out.flatten(1,-1))
        logvar = self.logvar(encoder_out.flatten(1,-1))
        std = torch.exp(logvar / 2)
        code = self.reparameterize(mu, logvar)
        
        out = self.linear2(code)

        x_hat = self.decoder(out.view(encoder_out.size()))

        return code, x_hat, mu, std

class VariationalAutoEncoderRawData(VariationalAutoEncoder):
    def __init__(self,inplanes:int=32):
        super().__init__()
        def building_blocks(in_dim, out_dim, filter_size=3,stride=1,padding=1):
            return nn.Sequential(nn.Conv2d(in_dim, out_dim, filter_size, stride=stride, padding=padding),
                                 nn.InstanceNorm2d(out_dim),
                                 nn.LeakyReLU(0.02))
        
        def building_blocks_trans(in_dim, out_dim, filter_size=3,stride=1,padding=0):
            return nn.Sequential(nn.ConvTranspose2d(in_dim, out_dim, filter_size, stride=stride, padding=padding),
                                 nn.InstanceNorm2d(out_dim),
                                 nn.LeakyReLU(0.02))
        self.encoder = nn.Sequential(
                            building_blocks(1,inplanes,3,1,1),
                            nn.MaxPool2d(2,2), #[batch,inplanes,96,96]

                            building_blocks(inplanes,inplanes*2,3,1,1),
                            nn.MaxPool2d(2,2), #[batch,inplanes*2,48,48]

                            building_blocks(inplanes*2,inplanes*4,3,1,1),
                            nn.MaxPool2d(2,2), #[batch,inplanes*4,24,24]

                            building_blocks(inplanes*4,inplanes*4,3,1,1),
                            nn.MaxPool2d(2,2), #[batch,inplanes*4,12,12]

                            building_blocks(inplanes*4,inplanes*8,3,1,1),
                            nn.MaxPool2d(2,2), #[batch,inplanes*4,6,6]
            
                            building_blocks(inplanes*8,inplanes*8,3,1,1),
                            nn.MaxPool2d(2,2), #[batch,inplanes*8,3,3]

                            )
        
        self.mu = nn.Sequential(
                            nn.Linear((inplanes*8*3*3), 2)
                            )
        
        self.logvar = nn.Sequential(
                            nn.Linear((inplanes*8*3*3), 2)
                            )

        self.linear2 = nn.Sequential(
                            nn.Linear(2, (inplanes*8*3*3))
                            )

        self.decoder = nn.Sequential(
                            nn.UpsamplingNearest2d(scale_factor=2),
                            building_blocks_trans(inplanes*8,inplanes*8,3,1,1),
            
                            nn.UpsamplingNearest2d(scale_factor=2),
                            building_blocks_trans(inplanes*8,inplanes*4,3,1,1),

                            nn.UpsamplingNearest2d(scale_factor=2),
                            building_blocks_trans(inplanes*4,inplanes*4,3,1,1),

                            nn.UpsamplingNearest2d(scale_factor=2),
                            building_blocks_trans(inplanes*4,inplanes*2,3,1,1),

                            nn.UpsamplingNearest2d(scale_factor=2),
                            building_blocks_trans(inplanes*2,inplanes,3,1,1),

                            nn.UpsamplingNearest2d(scale_factor=2),
                            nn.Conv2d(inplanes,1,3,1,1),
                            # nn.Sigmoid()
                            # nn.Tanh()
                            )

class VariationalAutoEncoderRadial2D(VariationalAutoEncoder):
    def __init__(self,inplanes:int=32):
        super().__init__()
        def building_blocks(in_dim, out_dim, filter_size=3,stride=1,padding=1):
            return nn.Sequential(nn.Conv2d(in_dim, out_dim, filter_size, stride=stride, padding=padding),
                                 nn.InstanceNorm2d(out_dim),
                                 nn.LeakyReLU(0.02))
        
        def building_blocks_trans(in_dim, out_dim, filter_size=3,stride=1,padding=0):
            return nn.Sequential(nn.ConvTranspose2d(in_dim, out_dim, filter_size, stride=stride, padding=padding),
                                 nn.InstanceNorm2d(out_dim),
                                 nn.LeakyReLU(0.02))

        self.encoder = nn.Sequential(
                            building_blocks(1,inplanes,3,1,1),
                            nn.MaxPool2d(2,2), #[batch,inplanes,192,80]

                            building_blocks(inplanes,inplanes*2,3,1,1),
                            nn.MaxPool2d(2,2), #[batch,inplanes*2,96,40]

                            building_blocks(inplanes*2,inplanes*4,3,1,1),
                            nn.MaxPool2d(2,2), #[batch,inplanes*4,48,20]

                            building_blocks(inplanes*4,inplanes*4,3,1,1),
                            nn.MaxPool2d(2,2), #[batch,inplanes*4,24,10]

                            building_blocks(inplanes*4,inplanes*8,3,1,1),
                            nn.MaxPool2d(2,2), #[batch,inplanes*4,12,5]
                            )
        
        self.mu = nn.Sequential(
                            nn.Linear((inplanes*8*12*5), 2)
                            )
        
        self.logvar = nn.Sequential(
                            nn.Linear((inplanes*8*12*5), 2)
                            )

        self.linear2 = nn.Sequential(
                            nn.Linear(2, (inplanes*8*12*5))
                            )

        self.decoder = nn.Sequential(
                            nn.UpsamplingNearest2d(scale_factor=2),
                            building_blocks_trans(inplanes*8,inplanes*4,3,1,1),

                            nn.UpsamplingNearest2d(scale_factor=2),
                            building_blocks_trans(inplanes*4,inplanes*4,3,1,1),

                            nn.UpsamplingNearest2d(scale_factor=2),
                            building_blocks_trans(inplanes*4,inplanes*2,3,1,1),

                            nn.UpsamplingNearest2d(scale_factor=2),
                            building_blocks_trans(inplanes*2,inplanes,3,1,1),

                            nn.UpsamplingNearest2d(scale_factor=2),
                            nn.Conv2d(inplanes,1,3,1,1),
                            # nn.Sigmoid()
                            nn.Tanh()
                            )

class VariationalAutoEncoderRadial1D(VariationalAutoEncoder):
    def __init__(self,inplanes:int=32):
        super().__init__()
        def building_blocks(in_dim, out_dim, filter_size=3,stride=1,padding=1):
            return nn.Sequential(nn.Conv1d(in_dim, out_dim, filter_size, stride=stride, padding=padding),
                                 nn.InstanceNorm1d(out_dim),
                                 nn.LeakyReLU(0.02))
        
        def building_blocks_trans(in_dim, out_dim, filter_size=3,stride=1,padding=0):
            return nn.Sequential(nn.ConvTranspose1d(in_dim, out_dim, filter_size, stride=stride, padding=padding),
                                 nn.InstanceNorm1d(out_dim),
                                 nn.LeakyReLU(0.02))

        self.encoder = nn.Sequential(
                            building_blocks(1,inplanes,3,1,1),
                            nn.MaxPool1d(2,2), #[batch,inplanes,80]

                            building_blocks(inplanes,inplanes*2,3,1,1),
                            nn.MaxPool1d(2,2), #[batch,inplanes*2,40]

                            building_blocks(inplanes*2,inplanes*4,3,1,1),
                            nn.MaxPool1d(2,2), #[batch,inplanes*4,20]

                            building_blocks(inplanes*4,inplanes*4,3,1,1),
                            nn.MaxPool1d(2,2), #[batch,inplanes*4,10]

                            building_blocks(inplanes*4,inplanes*8,3,1,1),
                            nn.MaxPool1d(2,2), #[batch,inplanes*4,5]
            
                            building_blocks(inplanes*8,inplanes*8,3,1,1),
                            building_blocks(inplanes*8,inplanes*16,3,1,1),
                            )
        
        self.mu = nn.Sequential(
                            nn.Linear((inplanes*16*4), (inplanes*4*4)),
                            nn.ReLU(),
                            nn.Linear((inplanes*4*4), (inplanes*4)),
                            nn.ReLU(),
                            nn.Linear((inplanes*4), 2)
                            )
        
        self.logvar = nn.Sequential(
                            nn.Linear((inplanes*16*4), (inplanes*4*4)),
                            nn.ReLU(),
                            nn.Linear((inplanes*4*4), (inplanes*4)),
                            nn.ReLU(),
                            nn.Linear((inplanes*4), 2)
                            )

        self.linear2 = nn.Sequential(
                            nn.Linear(2, (inplanes*4)),
                            nn.ReLU(),
                            nn.Linear((inplanes*4), (inplanes*4*4)),
                            nn.ReLU(),
                            nn.Linear((inplanes*4*4), (inplanes*16*4)),
                            )

        self.decoder = nn.Sequential(
                            building_blocks_trans(inplanes*16,inplanes*8,3,1,1),
                            building_blocks_trans(inplanes*8,inplanes*8,3,1,1),
            
                            nn.Upsample(scale_factor=2, mode='nearest'),
                            building_blocks_trans(inplanes*8,inplanes*4,3,1,1),

                            nn.Upsample(scale_factor=2, mode='nearest'),
                            building_blocks_trans(inplanes*4,inplanes*4,3,1,1),

                            nn.Upsample(scale_factor=2, mode='nearest'),
                            building_blocks_trans(inplanes*4,inplanes*2,3,1,1),

                            nn.Upsample(scale_factor=2, mode='nearest'),
                            building_blocks_trans(inplanes*2,inplanes,3,1,1),

                            nn.Upsample(scale_factor=2, mode='nearest'),
                            nn.Conv1d(inplanes,1,3,1,1),
                            # nn.Sigmoid()
                            # nn.Tanh()
                            )

class VariationalAutoEncoderMultiChannel1D(VariationalAutoEncoder):
    def __init__(self,inplanes:int=32):
        super().__init__()
        def building_blocks(in_dim, out_dim, filter_size=3,stride=1,padding=1):
            return nn.Sequential(nn.Conv1d(in_dim, out_dim, filter_size, stride=stride, padding=padding),
                                 nn.BatchNorm1d(out_dim),
                                 nn.LeakyReLU(0.02))
        
        def building_blocks_trans(in_dim, out_dim, filter_size=3,stride=1,padding=0):
            return nn.Sequential(nn.ConvTranspose1d(in_dim, out_dim, filter_size, stride=stride, padding=padding),
                                 nn.BatchNorm1d(out_dim),
                                 nn.LeakyReLU(0.02))

        self.encoder = nn.Sequential(
                            building_blocks(180,inplanes,3,1,1),
                            nn.MaxPool1d(2,2), #[batch,inplanes,64]

                            building_blocks(inplanes,inplanes,3,1,1),
                            nn.MaxPool1d(2,2), #[batch,inplanes*2,32]

                            building_blocks(inplanes,inplanes*2,3,1,1),
                            nn.MaxPool1d(2,2), #[batch,inplanes*4,16]

                            building_blocks(inplanes*2,inplanes*2,3,1,1),
                            nn.MaxPool1d(2,2), #[batch,inplanes*4,8]

                            building_blocks(inplanes*2,inplanes*4,3,1,1),
                            nn.MaxPool1d(2,2), #[batch,inplanes*4,4]
            
                            # building_blocks(inplanes*8,inplanes*8,3,1,1),
                            # building_blocks(inplanes*8,inplanes*16,3,1,1),
                            )
        
        self.mu = nn.Sequential(
                            nn.Linear((inplanes*4*4), (inplanes*2*4)),
                            nn.ReLU(),
                            nn.Linear((inplanes*2*4), (inplanes*4)),
                            nn.ReLU(),
                            nn.Linear((inplanes*4), 2)
                            )
        
        self.logvar = nn.Sequential(
                            nn.Linear((inplanes*4*4), (inplanes*2*4)),
                            nn.ReLU(),
                            nn.Linear((inplanes*2*4), (inplanes*4)),
                            nn.ReLU(),
                            nn.Linear((inplanes*4), 2)
                            )

        self.linear2 = nn.Sequential(
                            nn.Linear(2, (inplanes*4)),
                            nn.ReLU(),
                            nn.Linear((inplanes*4), (inplanes*2*4)),
                            nn.ReLU(),
                            nn.Linear((inplanes*2*4), (inplanes*4*4)),
                            )

        self.decoder = nn.Sequential(
#                             building_blocks_trans(inplanes*16,inplanes*8,3,1,1),
#                             building_blocks_trans(inplanes*8,inplanes*8,3,1,1),
            
                            nn.Upsample(scale_factor=2, mode='nearest'),
                            building_blocks_trans(inplanes*4,inplanes*2,3,1,1),

                            nn.Upsample(scale_factor=2, mode='nearest'),
                            building_blocks_trans(inplanes*2,inplanes*2,3,1,1),

                            nn.Upsample(scale_factor=2, mode='nearest'),
                            building_blocks_trans(inplanes*2,inplanes*1,3,1,1),

                            nn.Upsample(scale_factor=2, mode='nearest'),
                            building_blocks_trans(inplanes*1,inplanes,3,1,1),

                            nn.Upsample(scale_factor=2, mode='nearest'),
                            nn.Conv1d(inplanes,180,3,1,1),
                            # nn.Sigmoid()
                            # nn.Tanh()
                            )