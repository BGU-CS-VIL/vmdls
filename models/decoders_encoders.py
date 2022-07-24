import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.nn import functional as F


class MnistEncoder(nn.Module):
    def __init__(self,first_conv=False,max_pool1=False):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(28*28,128),
            nn.ReLU(),
            nn.Linear(128,64),
            nn.ReLU()
        )
    
    def forward(self,x):
        x = x.view(-1,784)
        return self.encoder(x)


class MnistDecoder(nn.Module):
    def __init__(self,latent_dim, input_height, first_conv =False, maxpool1=False):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim,64),
            nn.ReLU(),
            nn.Linear(64,128),
            nn.ReLU(),
            nn.Linear(128,input_height*input_height),
        )
    
    def forward(self,x):
        x = self.decoder(x)
        x = x.view(-1,1,28,28)
        return x



class MnistConvEncoder(nn.Module):
    def __init__(self,first_conv=False,max_pool1=False):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1,100,3,1,1),
            nn.ReLU(),
            nn.Conv2d(100,100,3,1,1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.Conv2d(100,100,3,1,1),
            nn.ReLU(),
            nn.Conv2d(100,100,3,1,1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.Conv2d(100,100,3,1,1),
            nn.ReLU(), 
            nn.MaxPool2d(2,2),           
        )
        self.dec_layer = nn.Linear(100*3*3,500)
    
    def forward(self,x):
        x = self.encoder(x)
        x = x.view(-1,100*3*3)
        return self.dec_layer(x)
        
    def get_layer_output(self, x,layer_num):
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        if layer_num == 1:
            x = self.encoder[0:3](x)
            return torch.flatten(self.avgpool(x),1)
        if layer_num == 2:
            x = self.encoder[0:5](x)
            return torch.flatten(self.avgpool(x),1)
        if layer_num == 3:
            x = self.encoder[0:7](x)
            return torch.flatten(self.avgpool(x),1)
        return self.forward(x)


class MnistConvDecoder(nn.Module):
    def __init__(self,latent_dim, input_height, first_conv =False, maxpool1=False):
        super().__init__()
        self.latent = nn.Linear(latent_dim,500)
        self.enc_layer = nn.Linear(500,100*3*3)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(100,100,2,1,0),
            nn.ReLU(),
            nn.ConvTranspose2d(100,100,2,1,0),
            nn.ReLU(),
            # # nn.Upsample(scale_factor=2),
            nn.ConvTranspose2d(100,100,2,1,0),
            nn.ReLU(),
            nn.ConvTranspose2d(100,100,2,1,0),
            nn.ReLU(),
            # # nn.Upsample(scale_factor=2),
            nn.ConvTranspose2d(100,100,2,2,0),
            nn.ReLU(),
            nn.ConvTranspose2d(100,100,2,2,0),
            nn.ReLU(),
            # # nn.Upsample(scale_factor=2),
            nn.Conv2d(100,1,1,1,0))        
    
    def forward(self,x):
        x= self.latent(x)
        x = self.enc_layer(x)
        x = x.view(-1,100,3,3)
        return self.decoder(x)


class CifarConvEncoder(nn.Module):
    def __init__(self,first_conv=False,max_pool1=False):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3,100,3,1,1),
            nn.ReLU(),
            nn.Conv2d(100,100,3,1,1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.Conv2d(100,100,3,1,1),
            nn.ReLU(),
            nn.Conv2d(100,100,3,1,1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.Conv2d(100,100,3,1,1),
            nn.ReLU(), 
            nn.MaxPool2d(2,2),           
        )
        self.dec_layer = nn.Linear(100*4*4,500)
    
    def forward(self,x):
        x = self.encoder(x)
        x = x.view(-1,100*4*4)
        return self.dec_layer(x)



class CifarConvDecoder(nn.Module):
    def __init__(self,latent_dim, input_height, first_conv =False, maxpool1=False):
        super().__init__()
        self.latent = nn.Linear(latent_dim,500)
        self.enc_layer = nn.Linear(500,100*4*4)
        self.decoder = nn.Sequential(
            nn.Conv2d(100,100,3,1,1),
            nn.ReLU(),
            nn.Conv2d(100,100,3,1,1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(100,100,3,1,1),
            nn.ReLU(),
            nn.Conv2d(100,100,3,1,1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(100,3,3,1,1),
            nn.Upsample(scale_factor=2))
        
    
    def forward(self,x):
        x= self.latent(x)
        x = self.enc_layer(x)
        x = x.view(-1,100,4,4)
        return self.decoder(x)