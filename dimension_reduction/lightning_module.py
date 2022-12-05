import random
import math
import torch
import numpy as np
import pytorch_lightning as pl
import matplotlib.pyplot as plt
from typing import Dict, Optional
from torch import nn
from torch.nn import functional as F
from .model import VariationalAutoEncoder


class VAE(pl.LightningModule):
    def __init__(self, 
                 model: nn.Module = VariationalAutoEncoder,
                 inplanes: int = 32, 
                 optimizer: str = 'adam',
                 learning_rate: float = 1e-4,
                 decay: float = 0.0,
                 lr_scheduler_kw: Optional[Dict] = None
                 ):

        super().__init__()
        self.save_hyperparameters()
        self.inplanes = inplanes
        self.model = model(inplanes=self.inplanes)
        
        # for gaussian likelihood
        self.log_scale = nn.Parameter(torch.Tensor([0.0]))
        self.log_scale_pe = nn.Parameter(torch.Tensor([0.0]))
        
        # optimizer
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.decay = decay
        self.lr_scheduler_kw = lr_scheduler_kw

        # latent representation
        self.latent = list()
    
    def binary_cross_entropy(self, x_hat, x):
        bin_loss = F.binary_cross_entropy_with_logits(x_hat, x, reduction='none')
        if len(x.size())==4:
            return bin_loss.sum(dim=(1, 2, 3))
        elif len(x.size())==3:
            return bin_loss.sum(dim=(1,2))
    
    def gaussian_likelihood(self, x_hat, logscale, x):
        scale = torch.exp(logscale)
        mean = x_hat
        dist = torch.distributions.Normal(mean, scale)

        # measure prob of seeing image under p(x|z)
        log_pxz = dist.log_prob(x)

        # normalisation factor that normalise the max likelihood to 1
        normalisation_factor = torch.log(math.sqrt(2*math.pi)*scale)
        log_pxz += normalisation_factor

        if len(x.size())==4:
            return log_pxz.sum(dim=(1, 2, 3))
        elif len(x.size())==3:
            return log_pxz.sum(dim=(1,2))
    
    def kl_divergence(self, z, mu, std):
        # --------------------------
        # Monte carlo KL divergence
        # --------------------------
        # 1. define the first two probabilities (in this case Normal for both)
        p = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(std))
        q = torch.distributions.Normal(mu, std)

        # 2. get the probabilities from the equation
        log_qzx = q.log_prob(z)
        log_pz = p.log_prob(z)

        # sum over last dim to go from single dim distribution to multi-dim
        kl = (log_qzx - log_pz)
        kl = kl.sum(-1)
        return kl

    # def on_train_start(self):
    #     self.logger.log_hyperparams(self.hparams, {"hp/precision": 0, "hp/recall": 0, "hp/f1":0})

    def training_step(self, train_batch, batch_idx):
        x = train_batch

        z, x_hat, mu, std = self.model(x)

        # reconstruction loss
        # recon_loss = self.gaussian_likelihood(x_hat, self.log_scale, x)
        recon_loss = self.binary_cross_entropy(x_hat,x)

        # kl
        kl = self.kl_divergence(z, mu, std)* 0.01
        
        # elbo
        # elbo = kl - recon_loss
        elbo = kl + recon_loss
        elbo = elbo.mean()

        metrics = {
            'loss':elbo,
            'train_kl_loss': kl.mean(),
            'train_recon_loss': recon_loss.mean(),
        }

        self.log('elbo', metrics['loss'], prog_bar=True, on_step=True)
        self.log('train_kl_loss', metrics['train_kl_loss'], prog_bar=True, on_step=True)
        self.log('train_recon_loss', metrics['train_recon_loss'], prog_bar=True, on_step=True)

        return metrics

    def training_epoch_end(self, training_step_outputs):
        epoch_train_loss = torch.stack([x['loss'] for x in training_step_outputs]).mean()
        epoch_train_kl_loss = torch.stack([x['train_kl_loss'] for x in training_step_outputs]).mean()
        epoch_train_recon_loss = -1*torch.stack([x['train_recon_loss'] for x in training_step_outputs]).mean()

        self.logger.experiment.add_scalar('Epoch_train_loss', epoch_train_loss,self.current_epoch)
        self.logger.experiment.add_scalar('Epoch_train_kl_loss', epoch_train_kl_loss,self.current_epoch)
        self.logger.experiment.add_scalar('Epoch_train_recon_loss', epoch_train_recon_loss,self.current_epoch)

    def validation_step(self, val_batch, batch_idx):
        x = val_batch
        z, x_hat, mu, std = self.model(x)

        # reconstruction loss
        # recon_loss = self.gaussian_likelihood(x_hat, self.log_scale, x)
        recon_loss = self.binary_cross_entropy(x_hat,x)
        
        # kl
        kl = self.kl_divergence(z, mu, std) * 0.01
        
        # elbo
        # elbo = kl - recon_loss
        elbo = kl + recon_loss
        elbo = elbo.mean()

        metrics = {
            'val_loss':elbo,
            'val_kl_loss': kl.mean(),
            'val_recon_loss': -1 * recon_loss.mean(),
        }

        metrics['x']=x
        metrics['x_hat']=x_hat

        self.log('val_loss', metrics['val_loss'], prog_bar=True, on_step=True)
        self.log('val_kl_loss', metrics['val_kl_loss'], prog_bar=True, on_step=True)
        self.log('val_recon_loss', metrics['val_recon_loss'], prog_bar=True, on_step=True)

        return metrics
    
    def validation_epoch_end(self, valid_step_outputs):
        epoch_val_loss = torch.stack([x['val_loss'] for x in valid_step_outputs]).mean()
        epoch_val_kl_loss = torch.stack([x['val_kl_loss'] for x in valid_step_outputs]).mean()
        epoch_val_recon_loss = torch.stack([x['val_recon_loss'] for x in valid_step_outputs]).mean()
        
        self.log('Epoch_val_loss', epoch_val_loss)
        self.logger.experiment.add_scalar('Epoch_val_kl_loss', epoch_val_kl_loss,self.current_epoch)
        self.logger.experiment.add_scalar('Epoch_val_recon_loss', epoch_val_recon_loss,self.current_epoch)

        # plot validation results
        last_batch = valid_step_outputs[-1]
        fig = _plot_evaluation(last_batch['x'], last_batch['x_hat'])
        fig.canvas.draw()

        # grab the pixel buffer and dump it into a numpy array
        eval_result = np.array(fig.canvas.renderer.buffer_rgba())
        eval_result = np.moveaxis(eval_result[:,:,:3],2,0)
        self.logger.experiment.add_image(f'eval_check_{self.current_epoch}', eval_result)

        # ap_dict = {}
        # for i, c in enumerate(ap_class):
        #     ap_dict[self.class_names[c]] = AP[i]
        # self.log_dict(ap_dict, on_step=False, on_epoch=True)

    def test_step(self, test_batch, batch_idx):
        x = test_batch
        _, _, embeddings, _= self.model(x)
        return embeddings
    
    def test_epoch_end(self, test_step_outputs):
        embeddings = torch.cat([x for x in test_step_outputs],dim=0)
        self.latent = embeddings.detach().cpu().numpy()

    def configure_optimizers(self):
        if self.optimizer == "adam":
            optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=self.learning_rate,
                weight_decay=self.decay)
        elif self.optimizer == "sgd":
            optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=self.learning_rate)
        else:
            raise Exception("Unknown optimizer. Only adam is implemented.")
        
        if self.lr_scheduler_kw != None:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, **self.lr_scheduler_kw)
            return dict(optimizer=optimizer, 
                        lr_scheduler=dict(scheduler=scheduler,
                                        frequency=1,
                                        monitor='Epoch_val_loss'))
        else:
            return optimizer

def _plot_evaluation(x, x_hat):
    imgs = x.to(device="cpu",dtype=torch.float32)
    targets = x_hat.to(device="cpu",dtype=torch.float32)
    
    batch_size = int(imgs.size(0))
    if batch_size > 4:
        img_ids = random.sample(range(batch_size),4)
        batch_size = 4
    else:
        img_ids = range(batch_size)
        
    fig, axs = plt.subplots(2,batch_size, figsize=(batch_size*1.5,3), dpi=150)
    for i in range(2):
        for j in range(batch_size):
            img_id = img_ids[j]
           
            axs_ = axs[i] if batch_size == 1 else axs[i, j]
            if i == 0:
                if len(imgs[img_id].shape) == 3:
                    axs_.imshow(imgs[img_id].squeeze().numpy())
                    axs_.axis("off")
                elif len(imgs[img_id].squeeze().shape) == 1: # for 1D profile
                    axs_.plot(imgs[img_id].squeeze().numpy())
                elif len(imgs[img_id].squeeze().shape) == 2: # for 1D multi-channel profile
                    img_ = imgs[img_id].squeeze().numpy().sum(axis=0)
                    axs_.plot(img_/img_.max())
            else:
                if len(imgs[img_id].shape) == 3:
                    img_ = torch.sigmoid(targets[img_id]).squeeze().numpy()
                    axs_.imshow(img_)
                    axs_.axis("off")
                elif len(imgs[img_id].squeeze().shape) == 1: # for 1D profile
                    img_ = torch.sigmoid(targets[img_id]).squeeze().numpy()
                    axs_.plot(img_,c='r')
                elif len(imgs[img_id].squeeze().shape) == 2: # for 1D multi-channel profile
                    img_ = torch.sigmoid(targets[img_id]).squeeze().numpy().sum(axis=0)
                    axs_.plot(img_/img_.max(),c='r')
    fig.subplots_adjust(wspace=0.0, hspace=0.05)
    fig.tight_layout()
    return fig









