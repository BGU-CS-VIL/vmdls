import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.nn import functional as F



class VGGM(nn.Module):
    def __init__(self,*args,**kwargs):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3,64,3,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64,128,3,padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(), #5
            nn.MaxPool2d(2,2),
            # nn.Dropout(0.25),
            nn.Conv2d(128,256,3,padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256,256,3,padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(), #12
            nn.MaxPool2d(2,2),
            # nn.Dropout(0.25),
            nn.Conv2d(256,256,3,padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256,256,3,padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256,256,3,padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256,256,3,padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(), #25
            nn.MaxPool2d(2,2),
            # nn.Dropout(0.25),
            nn.Flatten(),
            nn.Linear(256*4*4,1024),
            nn.ReLU(),
            # nn.Dropout(0.5),
            nn.Linear(1024,1024),
            nn.ReLU(),
            # nn.Dropout(0.5),
        )

    
    def forward(self,x):
        return self.encoder(x)
    
    def get_layer_output(self, x,layer_num):
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        if layer_num == 1:
            x = self.encoder[0:5](x)
            return torch.flatten(self.avgpool(x),1)
        if layer_num == 2:
            x = self.encoder[0:12](x)
            return torch.flatten(self.avgpool(x),1)
        if layer_num == 3:
            x = self.encoder[0:25](x)
            return torch.flatten(self.avgpool(x),1)
        return self.forward(x)