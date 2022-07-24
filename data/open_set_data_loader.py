from warnings import warn

import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from data.data_transforms import data_transforms, TwoCropTransform
from torch.utils.data import Dataset, DataLoader, ConcatDataset, BatchSampler

from torchvision.datasets import MNIST, CIFAR10,CIFAR100, KMNIST, Omniglot, ImageFolder
from data.invert import Invert
from data.datasets import *


class KnownUnkownDataset(Dataset):

    def __init__(self,known_dataset,unknown_dataset,unknown_label):
        self.known_dataset= known_dataset
        self.unknown_dataset = unknown_dataset        
        self.unknown_len = len(self.unknown_dataset)
        self.unknown_label = unknown_label
        self.known_len = len(self.known_dataset)

    def __len__(self):
        # return len(self.known_dataset)
        return self.known_len + self.unknown_len

    def __getitem__(self,index):
        # return self.known_dataset.__getitem__(index)
        if index < self.known_len:
            return self.known_dataset.__getitem__(index)
        else:
            sample,label = self.unknown_dataset.__getitem__(index-self.known_len)
            return sample, torch.tensor(self.unknown_label)



class BalancedBatchSampler():
    """
    BatchSampler - from a MNIST-like dataset, samples n_classes and within these classes samples n_samples.
    Returns batches of size n_classes * n_samples
    Taken from https://github.com/adambielski/siamese-triplet
    """

    def __init__(self, labels, n_classes, n_samples):
        self.labels = labels
        self.labels_set = list(set(self.labels.numpy()))
        self.label_to_indices = {label: np.where(self.labels.numpy() == label)[0]
                                 for label in self.labels_set}
        for l in self.labels_set:
            np.random.shuffle(self.label_to_indices[l])
        self.used_label_indices_count = {label: 0 for label in self.labels_set}
        self.count = 0
        self.n_classes = n_classes
        self.n_samples = n_samples
        self.n_dataset = len(self.labels)
        self.batch_size = self.n_samples * self.n_classes
        

    def __iter__(self):
        self.count = 0
        while self.count + self.batch_size < self.n_dataset:
            classes = np.random.choice(self.labels_set, self.n_classes, replace=False)
            indices = []
            for class_ in classes:
                indices.extend(self.label_to_indices[class_][
                               self.used_label_indices_count[class_]:self.used_label_indices_count[
                                                                         class_] + self.n_samples])
                self.used_label_indices_count[class_] += self.n_samples
                if self.used_label_indices_count[class_] + self.n_samples > len(self.label_to_indices[class_]):
                    np.random.shuffle(self.label_to_indices[class_])
                    self.used_label_indices_count[class_] = 0
            yield indices
            self.count += self.n_classes * self.n_samples


class TestingBalancedBatchSampler():
    """
    TestingBalancedBatchSampler - from a MNIST-like dataset, samples n_classes and within these classes samples n_samples.
    Returns batches of size n_classes * n_samples
    Taken from https://github.com/adambielski/siamese-triplet
    """

    def __init__(self, labels, n_classes, n_samples):
        self.labels = labels
        self.labels_set = list(set(self.labels.numpy()))
        self.label_to_indices = {label: np.where(self.labels.numpy() == label)[0]
                                 for label in self.labels_set}
        # for l in self.labels_set:
        #     np.random.shuffle(self.label_to_indices[l])
        self.used_label_indices_count = {label: 0 for label in self.labels_set}
        self.count = 0
        self.n_classes = n_classes
        self.n_samples = n_samples
        self.n_dataset = len(self.labels)
        self.batch_size = self.n_samples * self.n_classes
        

    def __iter__(self):
        self.count = 0
        self.cur_class = 0
        while self.count + self.batch_size < self.n_dataset:
            classes = []
            for i in range(self.n_classes):
                classes.append(self.cur_class)
                self.cur_class += 1
                if self.cur_class == len(self.labels_set):
                    self.cur_class = 0
            classes = np.array(classes)
            indices = []
            for class_ in classes:
                indices.extend(self.label_to_indices[class_][
                               self.used_label_indices_count[class_]:self.used_label_indices_count[
                                                                         class_] + self.n_samples])
                self.used_label_indices_count[class_] += self.n_samples
                if self.used_label_indices_count[class_] + self.n_samples > len(self.label_to_indices[class_]):
                    np.random.shuffle(self.label_to_indices[class_])
                    self.used_label_indices_count[class_] = 0
            yield indices
            self.count += self.n_classes * self.n_samples

    def __len__(self):
        return self.n_dataset // self.batch_size

def get_dict(class_num):   
    return {
        'mnist': MNIST,
        'kmnist': KMNIST,
        'omniglot':OmniglotWrapper,
        'mnist_noise':MNISTNoise,
        'noise':NOISE,
        'cifar_known':lambda root,train,download,transform, **kwargs: CIFAR10Partial(root,train,True,download,transform,class_num),
        'cifar_unknown':lambda root,train,download,transform, **kwargs: CIFAR10Partial(root,train,False,download,transform,class_num),
        'cifar100_known':lambda root,train,download,transform, **kwargs: CIFAR100Partial(root,train,True,download,transform,class_num),
        'cifar100_unknown':lambda root,train,download,transform, **kwargs: CIFAR100Partial(root,train,False,download,transform,class_num),
        'imagenet-resize':lambda root,train,download,transform, **kwargs:ImageFolder("data/openset/Imagenet_resize", transform=transform),
        'imagenet-crop':lambda root,train,download,transform, **kwargs: ImageFolder("data/openset/Imagenet", transform=transform),
        'lsun-resize':lambda root,train,download,transform, **kwargs: ImageFolder("data/openset/LSUN_resize", transform=transform),
        'lsun-crop':lambda root,train,download,transform, **kwargs: ImageFolder("data/openset/LSUN", transform=transform)
    }    



class KnownUnknownDataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: str = "./",
        num_workers: int = 16,
        batch_size: int = 2500,
        known_dataset: str = 'mnist',
        unknown_dataset: str = 'kmnist',
        data_transforms: str = 'mnist',
        class_num: int = 10,
        is_testing: bool = False,
        batch_classes: int = 10,
        sample_count: int = 40,
        *args,
        **kwargs,
    ):
        
        super().__init__(*args, **kwargs)
        self.dims = (1, 28, 28)
        self.data_dir = data_dir
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.class_num = class_num
        self.known_dataset = known_dataset
        self.unknown_dataset = unknown_dataset
        self.data_transforms = data_transforms
        self.is_known = True
        self.is_testing = is_testing
        self.batch_classes = batch_classes
        self.sample_count = sample_count



    @property
    def num_classes(self):
        return self.class_num

    # def prepare_data(self):

    def train_dataloader(self):
        """
        MNIST train set removes a subset to use for validation
        Args:
            batch_size: size of batch
            transforms: custom transforms
        """
        self.is_known = True
        datasets_dict= get_dict(self.class_num)
        train_dataset = datasets_dict[self.known_dataset](root=self.data_dir,
                                                    train=True,
                                                    download=True,
                                                    transform = data_transforms[self.data_transforms]['train'])
                # train_dataset = datasets_dict[self.known_dataset](root=self.data_dir,
                #                                     train=True,
                #                                     download=True,
                #                                     transform = TwoCropTransform(data_transforms[self.data_transforms]['train']))                        
        if self.is_testing == False: #Hacky cause PL is annoying
            batch_sampler = BalancedBatchSampler(train_dataset.targets,self.batch_classes,self.sample_count)
            loader = DataLoader(
                train_dataset,
                batch_sampler = batch_sampler,
                # batch_size=self.batch_size,
                # shuffle=False,
                num_workers=self.num_workers,
                # drop_last=False,
                pin_memory=True,
            )
        else:
            batch_sampler = TestingBalancedBatchSampler(train_dataset.targets,self.batch_classes,self.sample_count)
            loader = DataLoader(
                train_dataset,
                batch_sampler = batch_sampler,
                # batch_size=self.batch_size,
                # shuffle=False,
                num_workers=self.num_workers,
                # drop_last=False,
                pin_memory=True)
            # loader = DataLoader(
            #     train_dataset,
            #     # batch_sampler = batch_sampler,
            #     batch_size=self.batch_size,
            #     # shuffle=False,
            #     num_workers=self.num_workers,
            #     drop_last=False,
            #     pin_memory=True
            # )
        return loader

    def val_dataloader(self):
        self.is_known = True
        datasets_dict= get_dict(self.class_num)
        test_known = datasets_dict[self.known_dataset](root=self.data_dir,
                                                    train=False,
                                                    download=True,
                                                    transform = data_transforms[self.data_transforms]['val'])
        self.is_known = False
        test_unknown = datasets_dict[self.unknown_dataset](root=self.data_dir,
                                                    train=False,
                                                    download=True,
                                                    transform = data_transforms[self.data_transforms]['val'])
        test_dataset = KnownUnkownDataset(test_known,test_unknown,self.class_num)
        self.is_known = True
        loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=False,
            pin_memory=True,
        )
        return loader

    def test_dataloader(self):
        datasets_dict= get_dict(self.class_num)
        test_known = datasets_dict[self.known_dataset](root=self.data_dir,
                                                    train=False,
                                                    download=True,
                                                    transform = data_transforms[self.data_transforms]['val'])
        self.is_known = False
        test_unknown = datasets_dict[self.unknown_dataset](root=self.data_dir,
                                                    train=False,
                                                    download=True,
                                                    transform = data_transforms[self.data_transforms]['val'])
        
        test_dataset = KnownUnkownDataset(test_known,test_unknown,self.class_num)
        self.is_known = True
        loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=False,
            pin_memory=True,
        )
        return loader

