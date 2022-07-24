from warnings import warn

import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, ConcatDataset

from torchvision.datasets import MNIST, CIFAR10,CIFAR100, KMNIST, Omniglot, LSUN
from data.invert import Invert
import numpy as np
from PIL import Image
import os

class CIFAR10Partial(CIFAR10):
    def __init__(self, root,train,known,download,transform,class_num=10):
        super(CIFAR10Partial, self).__init__(root=root, train=train,transform=transform,download=download)
        self.targets = torch.tensor(np.array(self.targets))  
        if known:
            self.data = self.data[self.targets < class_num]
            self.targets = self.targets[self.targets < class_num]
        else:
            self.data = self.data[self.targets >= class_num]
            self.targets = self.targets[self.targets >= class_num]

    def __getitem__(self, index):
        img,target = super(CIFAR10Partial,self).__getitem__(index)
        return img, target


class CIFAR100Partial(CIFAR100):
    def __init__(self, root,train,known,download,transform,class_num=20):
        super(CIFAR100Partial, self).__init__(root=root, train=train,transform=transform,download=download)
        self.targets = torch.tensor(np.array(self.targets))
        # if train == True:
        #     self.targets = np.load('data/cifar_coarse/coarse_train.npy')
        # else:
        #     self.targets = np.load("data/cifar_coarse/coarse_test.npy")     
        if known:
            self.data = self.data[self.targets < class_num]
            self.targets = self.targets[self.targets < class_num]
        else:
            self.data = self.data[self.targets >= class_num]
            self.targets = self.targets[self.targets >= class_num]

    def __getitem__(self, index):
        img,target = super(CIFAR100Partial,self).__getitem__(index)
        return img, target        
        
class OmniglotWrapper(Omniglot):
    def __init__(self, root,train, **kwargs):
        super(OmniglotWrapper, self).__init__(root=root, background=train, **kwargs)
        self.transform = transforms.Compose([Invert(),transforms.Resize((28,28)),self.transform])
        self._flat_character_images=self._flat_character_images[:10000]


class ImageNetUnknown(Dataset):
    def __init__(self, root, txt, transform=None,set_size=10000,type='resize', **kwargs):
        
        self.img_path = []
        self.labels = []
        if type == 'resize':
            t = transforms.Resize((32,32))
        else:
            t = transforms.CenterCrop((32,32))
        self.transform = transforms.Compose([t,transform])
        with open(txt) as f:
            for line in f:
                self.img_path.append(os.path.join(root, line.split()[0]))
                self.labels.append(int(line.split()[1]))
        self.perm = np.random.permutation(len(self.labels)).tolist()
        self.labels = np.array(self.labels)
        self.img_path = np.array(self.img_path)
        self.labels = self.labels[self.perm[:set_size]]
        self.img_path = self.img_path[self.perm[:set_size]]


    def __len__(self):
        return len(self.labels)
        
    def __getitem__(self, index):
        path = self.img_path[index]
        label = self.labels[index]        
        with open(path, 'rb') as f:
            sample = Image.open(f).convert('RGB')
        
        if self.transform is not None:
            sample = self.transform(sample)

        return sample, label



class LSUNUnknown(LSUN):
    def __init__(self, root,transform=None,set_size=10000,type='resize', **kwargs):
        super(LSUNUnknown, self).__init__(root, 'train',transform=None)
        if type == 'resize':
            t = transforms.Resize((32,32))
        else:
            t = transforms.CenterCrop((32,32))
        self.transforms = transforms.Compose([t,transform])
        self.set_size=set_size
        self.choosen_indices = np.random.permutation(self.length)[0:self.set_size]


    def __len__(self):
        return self.set_size

    def __getitem__(self,index):
        sample,label = super(LSUNUnknown, self).__getitem__(self.choosen_indices[index])
        return self.transforms(sample), label


class MNISTNoise(MNIST):
    def __init__(self,root,train,download,transform):
        super(MNISTNoise, self).__init__(root= root, train = train,download = download,transform=transform)

    def __getitem__(self,index):
        sample,label = super(MNISTNoise, self).__getitem__(index)
        noise = torch.from_numpy(np.random.rand(1, 28, 28)).float()
        sample += noise
        return sample, label


class NOISE(Dataset):
    def __init__(self,root,train,download,transform):
        self.len = 50000 if train else 10000

    def __len__(self):
        return self.len

    def __getitem__(self,index):        
        noise = torch.from_numpy(np.random.rand(1, 28, 28)).float()
        return noise, 10
