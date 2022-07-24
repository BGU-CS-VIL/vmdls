from argparse import ArgumentParser

import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.optim.lr_scheduler import StepLR,CyclicLR
from pytorch_metric_learning import losses,miners,distances
from torchvision.transforms.transforms import RandomRotation
from losses.modded_triplets import TripletMarginLossModded
from pytorch_lightning.callbacks import Callback
from sklearn.decomposition import PCA
from pl_bolts.models.autoencoders import AE
from models.mnist_ae import MNIST_AE
# import seaborn as sns
import pandas as pd
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
try:
    from tsnecuda import TSNE
except:
    from sklearn.manifold import TSNE
import numpy as np
from pytorch_lightning.loggers.neptune import NeptuneLogger
from sklearn.metrics import f1_score,auc
from models.decoders_encoders import *
from tqdm import tqdm
from neptunecontrib.api import log_table
from losses.losses import SupConLoss
from losses.mmd import compute_mmd
from PIL import Image
from torchvision.models import vgg11
import kornia
from torchvision import transforms
from models.densenet import *
from models.vgg import VGGM

from pl_bolts.datamodules import CIFAR10DataModule
from models.wide_resnet import WideResNet
import matplotlib.ticker as ticker


class UpdateMargin(Callback):
    def __init__(self,margin_max_distance, margin_jumps,margins_epoch):
        self.margin_max_distance = margin_max_distance
        self.margin_jumps = margin_jumps
        self.margins_epoch = margins_epoch

    def on_train_epoch_start(self,trainer, pl_module):
        if trainer.current_epoch >= self.margins_epoch and pl_module.metric_loss.neg_margin <= self.margin_max_distance:
            pl_module.metric_loss.neg_margin +=self.margin_jumps



class InitializeGaussians(Callback):
    def on_test_epoch_start(self,trainer,pl_module):
        pl_module.eval()
        if trainer.current_epoch == 0:
            pl_module.initialize_gaussians()




from models.resnet_encoders import (
    resnet18_decoder,
    resnet18_encoder,
    resnet50_decoder,
    resnet50_encoder,
)

ae_dict = {
    'cifar_known':(lambda:AE(input_height=32), 'https://pl-bolts-weights.s3.us-east-2.amazonaws.com/ae/ae-cifar10/checkpoints/epoch%3D96.ckpt'),
    'cifar100_known':(lambda:AE(input_height=32,enc_type='resnet50'), 'models/ae_cifar100.ckpt'),
    'mnist':(lambda:MNIST_AE(input_height=28,enc_type='mnist'),'models/ae_mnist.ckpt')
}



class GMM_VAE_Contrastive(pl.LightningModule):   
    def __init__(
        self,
        input_height: int,
        enc_type: str = 'resnet18',
        first_conv: bool = False,
        maxpool1: bool = False,
        enc_out_dim: int = 512,
        kl_coeff: float = 0.1,
        latent_dim: int = 256,
        lr: float = 1e-4,
        class_num: int = 10,
        generation_epoch: int = 50,
        lower_bound: float = 0.05,
        step_size: int = 100,
        step_gamma: float = 0.1,
        cov_scaling: float = 5.0,
        recon_weight: float = 0.1,
        gen_weight: float = 5.0,
        log_tsne:bool = False,
        weights:str = None,
        finetune:bool = False,
        finetune_lr:float = 0.001,
        known_dataset:str = 'cifar_known',
        opt:str ='sgd',
        ae_features: bool= False,
        margin_max_distance: float = 32,
        sample_count: int = 40,
        **kwargs
    ):

        super(GMM_VAE_Contrastive, self).__init__()

        self.save_hyperparameters()
        self.known_dataset = known_dataset
        self.lr = lr
        self.step_size = step_size
        self.step_gamma = step_gamma
        self.kl_coeff = kl_coeff
        self.enc_out_dim = enc_out_dim
        self.latent_dim = latent_dim
        self.input_height = input_height
        self.class_num = class_num
        self.generation_epoch = generation_epoch
        self.cov_scaling = cov_scaling
        self.log_tsne = log_tsne
        self.is_tested = 0
        self.gen_weight = gen_weight
        self.weights = weights
        self.finetune = finetune
        self.finetune_lr = finetune_lr
        self.opt = opt
        self.ae_features = ae_features
        self.margin_max_distance = margin_max_distance
        self.sample_count = sample_count


        valid_encoders = {
            'resnet18': {'enc': resnet18_encoder, 'dec': resnet18_decoder},
            'resnet50': {'enc': resnet50_encoder, 'dec': resnet50_decoder},
            'mnist':{'enc':MnistEncoder, 'dec':MnistDecoder},
            'mnist-conv':{'enc':MnistConvEncoder, 'dec':MnistConvDecoder},
            'cifar':{'enc':CifarConvEncoder, 'dec':CifarConvDecoder},
            'densenet':{'enc':DenseNet3,'dec':DenseNet3},
            'vgg':{'enc':VGGM,'dec':VGGM},
            'wresnet':{'enc':WideResNet,'dec':WideResNet}
        }

        if enc_type not in valid_encoders:
            self.encoder = resnet18_encoder(first_conv, maxpool1)
        elif enc_type == 'densenet':
            self.encoder = DenseNet3(100,class_num)            
            print(self.encoder.in_planes)
        elif enc_type == 'wresnet':
            self.encoder = WideResNet(28,class_num,10,0.1)
        else:
            self.encoder = valid_encoders[enc_type]['enc'](first_conv, maxpool1)
        if self.weights != None:
            pretrained_dict = torch.load(self.weights)['state_dict']
            model_dict = self.encoder.state_dict()
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict) 
            self.encoder.load_state_dict(pretrained_dict)
        self.fc_mu = nn.Linear(self.enc_out_dim, self.latent_dim)
        self.fc_var = nn.Linear(self.enc_out_dim, self.latent_dim)
        self.metric_loss = TripletMarginLossModded(margin=0.1,neg_margin=0.15)
        self.recon_weight = recon_weight
        self.precentiles = []
        self.class_gaussians = []
        self.lower_bound = lower_bound
        self.generated_count = 0
        self.seen_test = False
        self.class_gaussian = []
        
        self.ae = ae_dict['cifar_known'][0]()
        self.ae = self.ae.load_from_checkpoint(ae_dict['cifar_known'][1])

        self.ae.freeze()
        self.sift = kornia.feature.SIFTDescriptor(self.input_height)

        self.rand_transforms = transforms.Compose([
            transforms.RandomCrop(input_height, padding=4),
            transforms.RandomRotation(10),
            transforms.RandomHorizontalFlip()])
        if self.finetune:
            for param in self.encoder.parameters():
                param.requires_grad = False
        self.enc_type = enc_type


    @staticmethod
    def pretrained_weights_available():
        return list(GMM_VAE_Contrastive.pretrained_urls.keys())

    def from_pretrained(self, checkpoint_name):
        if checkpoint_name not in GMM_VAE_Contrastive.pretrained_urls:
            raise KeyError(str(checkpoint_name) + ' not present in pretrained weights.')

        return self.load_from_checkpoint(GMM_VAE_Contrastive.pretrained_urls[checkpoint_name], strict=False)

    def forward(self, x):
        x = self.encoder(x)
        # return x
        mu = self.fc_mu(x)
        log_var = self.fc_var(x)
        p, q, z = self.sample(mu, log_var)
        return z

    def _run_step(self, x):
        x = self.encoder(x)
        # return x
        mu = self.fc_mu(x)
        log_var = self.fc_var(x)
        p, q, z = self.sample(mu, log_var)
        # return z,q
        return z, mu, log_var, q

    def sample(self, mu, log_var):
        std = torch.exp(log_var / 2)
        p = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(std))
        q = torch.distributions.Normal(mu, std)
        z = q.rsample()
        return p, q, z

    def filter_samples(self,all_samples,labels):
        for i in range(self.class_num):
            probs = self.class_gaussian[i].log_prob(all_samples) < self.class_bounds[i]      
            all_samples = all_samples[probs]
            labels = labels[probs]
        return all_samples,labels

    def filter_just_samples(self,samples):
        for i in range(self.class_num):
            probs = self.class_gaussian[i].log_prob(samples) < self.class_bounds[i]      
            samples = samples[probs]
        return samples




    def step(self, batch, batch_idx):
        x, y = batch
        cur_classes = torch.unique(y).long()

        z ,mu,log_var, q= self._run_step(x)        

        kl = 0
        log_qz = q.log_prob(z)        
        for i in cur_classes:
            
            y_count = (y==i).sum()
            rel_zs = z[y==i,:]
            self.class_temp_means[i,:] += rel_zs.sum(axis = 0).detach()
            self.class_temp_cov[i,:,:] += torch.matmul(rel_zs.T,rel_zs).detach()
            self.class_counts[i] += y_count.detach()                       
            #Estimation:
            cp = torch.distributions.Normal(self.class_means[i,:].repeat(y_count,1), torch.ones_like(rel_zs)*(0.1 if i < self.class_num else 1))
            log_pz = cp.log_prob(rel_zs)
            kl_class = log_qz[y==i] - log_pz
            kl_class = kl_class.mean()
            kl_class *= self.kl_coeff
            kl += kl_class / len(cur_classes)
            #Analytical
            # rel_mu = mu[y==i,:]
            # rel_var = torch.exp(log_var[y==i,:]/2)
            # q = torch.distributions.Normal(rel_mu, rel_var)
            # p = torch.distributions.Normal(self.class_means[i,:].repeat(y_count,1), torch.ones_like(rel_zs)*(0.1 if i < self.class_num else 1))
            # kl_loss = torch.distributions.kl.kl_divergence(q,p).mean()*self.kl_coeff
            # kl += kl_loss / len(cur_classes)
            if self.trainer.current_epoch > 0:
                class_ll = self.class_gaussian[i].log_prob(rel_zs).detach()
                self.precentiles[i].append(class_ll)

        contrastive_loss = self.metric_loss(z,y)
        if self.trainer.current_epoch < self.generation_epoch:
            kl = 0
        loss = contrastive_loss + kl
        logs = {
            "loss": loss,
            "kl": kl,
            "triplet_loss": contrastive_loss,
        }
        return loss, logs




    def test_step(self, batch, batch_idx):
        x, y = batch
        mu= self.fc_mu(self.encoder(x))
        z = mu
        l_features = [torch.flatten(self.encoder.get_layer_output(x,i),1) for i in range(1,5)] + [mu]
        l_features = torch.hstack(l_features)
        # l_features = torch.hstack([l_features,mu])
        # z ,x_hat,p, q= self._run_step(x_hat)
        # if self.is_tested == 1:
            # z2 ,x_hat2 ,p, q= self._run_step(x_hat)
            # x_hat[y==self.class_num] = x_hat2[y==self.class_num]
            # z[y==self.class_num] = z2[y==self.class_num]
        cur_classes = torch.unique(y)     
        if self.trainer.testing and self.is_tested == 0:
            for i in cur_classes:
                y_count = (y==i).sum()
                rel_zs = z[y==i,:]
                rel_l_features = l_features[y==i,:]
                self.class_temp_means[i,:] += rel_zs.sum(axis = 0).detach()
                self.class_temp_cov[i,:,:] += torch.matmul(rel_zs.T,rel_zs).detach()
                self.class_counts[i] += y_count.detach()
                self.early_temp_means[i,:] = rel_l_features.sum(axis = 0).detach()
                self.early_temp_cov[i,:,:] = torch.matmul(rel_l_features.T,rel_l_features).detach()

        # x_hat = x + (0.01)*torch.randn(x.shape,device=self.device)
        # x_hat = self.rand_transforms(x)
        x_hat=x
        if self.is_tested != 0:
            xshape = x.shape[1]
            chunk_size = int(x.shape[0]/4)
            x_hat = []
            for i in range(4):
                x_hat.append(self.ae(x[chunk_size*i:chunk_size*(i+1)]))
            x_hat = torch.vstack(x_hat)
            desc_x_encoder = torch.flatten(self.encoder.get_layer_output(x,1),1)
            desc_x_hat_encoder = torch.flatten(self.encoder.get_layer_output(x_hat,1),1)
            desc_x = desc_x_encoder
            desc_x_hat = desc_x_hat_encoder
        else:
            desc_x = 0
            desc_x_hat = 0
            x_hat = x       
        return z,y,x_hat,x,desc_x,desc_x_hat,l_features

    def training_epoch_end(self, outputs):
        class_gaussians = []
        bad_classes = 0
        for i in range(self.class_num):
            if self.trainer.current_epoch > 0:
                self.precentiles[i] = torch.hstack(self.precentiles[i])
                self.class_bounds[i] = torch.quantile(self.precentiles[i],self.lower_bound)
            self.class_means[i,:] = self.class_temp_means[i,:]/self.class_counts[i]
            self.class_cov[i,:,:] = self.class_temp_cov[i,:,:]/self.class_counts[i] - torch.matmul(self.class_means[i,:].view(1,-1).T,self.class_means[i,:].view(1,-1)) 
            try:
                dist = torch.distributions.multivariate_normal.MultivariateNormal(self.class_means[i,:],self.class_cov[i,:,:])
            except:
                bad_classes += 1
                dist = torch.distributions.multivariate_normal.MultivariateNormal(torch.zeros_like(self.class_means[i,:],device=self.device),torch.eye(self.class_cov[i,:,:].shape[0],device=self.device))           
            class_gaussians.append(dist)

        cur_num = self.class_num 
        self.full_gaussian_mean = self.class_means.sum(axis=0) / self.class_counts.sum()
        self.full_gaussian_cov = self.class_temp_cov.sum(axis=0) / self.class_counts.sum() - torch.matmul(self.full_gaussian_mean.view(1,-1).T,self.full_gaussian_mean.view(1,-1)) 
        self.class_gaussian = class_gaussians
        self.precentiles = [[] for i in range(cur_num)]
        self.class_counts = torch.zeros(cur_num,device=self.device)
        self.class_temp_means = torch.zeros((cur_num,self.latent_dim),device=self.device)
        self.class_temp_cov = torch.zeros(((cur_num,self.latent_dim,self.latent_dim)),device=self.device)


        if self.trainer.current_epoch == 21 and self.finetune:
            print("Adding encoder parameters")
            for param in self.encoder.parameters():
                param.requires_grad = True
            self.trainer.accelerator_backend.setup_optimizers(self)



    def training_step(self, batch, batch_idx):
        loss, logs = self.step(batch, batch_idx)
        self.log_dict(
            {f"train_{k}": v for k, v in logs.items()}, on_step=True, on_epoch=False
        )
        return loss

    def validation_step(self, batch, batch_idx):
        z,y,x_hat,x,_,_,_ = self.test_step(batch, batch_idx)
        return z,y,x_hat,x

    def classify_data(self,data):
        parr = torch.zeros(data.shape[0],self.class_num)
        for i in range(self.class_num):
            probs = self.class_gaussian[i].log_prob(data)
            probs[probs < self.class_bounds[i]] = -10**10
            parr[:,i] = probs
        values,labels = torch.max(parr,dim=1)
        labels[values < -10**9] = self.class_num
        print(len(labels))
        print((labels == self.class_num).sum())
        return labels.to(self.device)


    def get_data_top_val(self,data,all_sift_distances):
        parr = torch.zeros(data.shape[0],self.class_num)
        for i in range(self.class_num):
            probs = self.class_gaussian[i].log_prob(data)
            parr[:,i] = probs
        max_num = parr.max()+1
        for i in range(self.class_num):
            probs = self.class_gaussian[i].log_prob(data)
            probs -= max_num
            parr[:,i] = probs*all_sift_distances
        # parr[:,self.class_num] /= 1000
        row_sum = torch.logsumexp(parr,1)
        values,labels = torch.max(parr,dim=1)
        pvals = torch.exp(values-row_sum)
        # values *=1-pvals
        return values,labels

    def get_fpr_tpr(self,values,pred_labels,labels,threshold):
        neg_samples_count = float(np.sum(labels == self.class_num))
        pos_samples_count = float(np.sum(labels < self.class_num))
        neg_vals = values[labels == self.class_num]
        pos_vals = values[labels < self.class_num]
        pos_samples_count_cor = float(np.sum(pred_labels[labels < self.class_num] == labels[labels < self.class_num]))
        fp_count = float(np.sum(neg_vals >= threshold))
        tp_count_cor = float(np.sum((pos_vals >= threshold)& (pred_labels[labels < self.class_num] == labels[labels < self.class_num])))
        tp_count = float(np.sum(pos_vals >= threshold))
        return fp_count/neg_samples_count,tp_count/pos_samples_count,tp_count_cor/pos_samples_count_cor

    
    def log_test_statistics(self,values,pred_labels,labels):
        print(np.min(values),np.max(values))
        thresholds = np.flip(np.arange(np.min(values),np.max(values),1.0))
        thresholds = np.flip(np.arange(-1000,0,1))
        f1_scores = []
        pos_acc = []
        neg_acc = []
        pos_labels = labels[labels < self.class_num]
        neg_labels = labels[labels == self.class_num]
        pos_pred_labels = pred_labels[labels < self.class_num]
        neg_pred_labels = pred_labels[labels == self.class_num]
        self.log(f"Acc Pos (no unkown det):", float(np.sum(pos_pred_labels == pos_labels)/len(pos_labels)))
        misclassification_as_openset = []
        for t in tqdm(thresholds):
            cur_preds = np.copy(pred_labels)
            cur_preds[values < t] = self.class_num
            pos_pred_labels = cur_preds[labels < self.class_num]
            neg_pred_labels = cur_preds[labels == self.class_num]
            f1_scores.append(f1_score(cur_preds,labels,average='macro'))
            pos_acc.append(float(np.sum(pos_pred_labels == pos_labels)/len(pos_labels)))
            neg_acc.append(float(np.sum(neg_pred_labels == neg_labels)/len(neg_pred_labels)))
            mistakes = float(np.sum(pos_pred_labels == self.class_num))/ float(np.sum(pos_pred_labels != pos_labels))
            misclassification_as_openset.append(mistakes)
        f1_scores = np.array(f1_scores)
        pos_acc = np.array(pos_acc)
        neg_acc = np.array(neg_acc)
        best_t = np.argmax(f1_scores)
        self.log(f"Best f1 score (threshold:{thresholds[best_t]}) mistakes_as_open:{misclassification_as_openset[best_t]}:", np.max(f1_scores))
        fig = plt.figure()
        plt.plot(thresholds, f1_scores, color='darkorange')
        plt.xlim([np.min(thresholds), np.max(thresholds)])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Threshold')
        plt.ylabel('F1')
        plt.title('F1 per threshold')
        self.trainer.logger.experiment.log_image('F1 Curve', fig, description=f'F1 Curve')
        df = pd.DataFrame()
        df['Threshold'] = thresholds
        df['F1'] = f1_scores
        df['Pos acc'] = pos_acc
        df['Neg acc'] = neg_acc
        log_table('Thresholds', df,experiment=self.trainer.logger.experiment)



    
    def plot_roc_curve(self,all_data,labels,all_sift_distances):
        all_data = torch.tensor(all_data,device=self.device)
        values,pred_labels = self.get_data_top_val(all_data,all_sift_distances)
        values = values.cpu().numpy()
        pred_labels = pred_labels.cpu().numpy()
        self.log_test_statistics(values,pred_labels,labels)
        thresholds = np.flip(np.geomspace(np.max(values),np.min(values),num=100000))
        fpr = []
        tpr = []
        aurocNew = 0.0
        fprTemp = 1.0
        fp=0
        tp=0
        tp_cor=0
        fpr_at_095 = 0
        tpr_at_095 = 0
        for t in tqdm(range(len(thresholds))):
            fp,tp,tp_cor = self.get_fpr_tpr(values,pred_labels,labels,thresholds[t])
            if tp >= 0.95:
                fpr_at_095 = fp
                tpr_at_095 = tp
            fpr.append(fp)
            tpr.append(tp)
            aurocNew += (-fp+fprTemp)*tp_cor
            fprTemp = fp
        aurocNew += fp * tp_cor
        self.log('CAUROC',aurocNew)
        self.log("tpr-fpr",fpr_at_095)
        self.log("tpr-at-095",tpr_at_095)
        fig = plt.figure()
        plt.plot(fpr, tpr, color='darkorange')
        plt.plot([0, 1], [0, 1], color='navy',linestyle='--')
        plt.xlim([-0.00099, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Roc Curve')
        self.trainer.logger.experiment.log_image('ROC', fig, description=f'ROC TEST')
        self.log('ROC-AUC-Score',auc(fpr,tpr))
        df = pd.DataFrame()
        df['Threshold'] = thresholds
        df['FPR'] = np.array(fpr)
        df['TPR'] = np.array(tpr)
        log_table('FPR_TPR', df,experiment=self.trainer.logger.experiment)

    def test_epoch_end(self, outputs):
        all_data = torch.vstack([x[0] for x in outputs]).cpu().numpy()
        all_labels = torch.hstack([x[1] for x in outputs]).cpu().numpy()
        all_known_data = all_data[all_labels < self.class_num,...]
        all_known_labels = all_labels[all_labels < self.class_num]
        all_early_features = torch.vstack([x[6] for x in outputs]).cpu().numpy()

        all_data = all_early_features
        all_known_data = all_data[all_labels < self.class_num,...]

        if self.log_tsne:
            x_te_proj_pca = TSNE(n_components=2, perplexity=30, learning_rate=200).fit_transform(all_known_data)
            x_te_proj_df = pd.DataFrame(x_te_proj_pca[:, :2], columns=['Proj1', 'Proj2'])
            x_te_proj_df['label'] = all_known_labels
            fig = plt.figure()
            ax = sns.scatterplot('Proj1', 'Proj2', data=x_te_proj_df,
                    palette='tab20',
                    hue='label',
                    linewidth=0,
                    alpha=0.6,
                    s=7)
            box = ax.get_position()
            ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
            ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

        
        
        if self.is_tested == 0:
            self.calculate_gaussian()
            self.is_tested = 1
            np.save("all_data_train.npy",all_data)
            np.save("all_labels_train.npy",all_labels)
            if self.log_tsne:
                self.trainer.logger.experiment.log_image('Train T-SNE', fig, description=f'Final TEST')
            return
        elif self.is_tested == 1:
            self.class_gaussian = self.early_gaussians

        if self.log_tsne:
            self.trainer.logger.experiment.log_image('T-SNE Without Unknown', fig, description=f'Final TEST')


        all_sift = torch.vstack([x[4].reshape(x[4].shape[0],-1) for x in outputs])
        all_sift_hat = torch.vstack([x[5].reshape(x[5].shape[0],-1) for x in outputs])
        all_sift_distances = ((all_sift-all_sift_hat)**2).mean(axis=1)
        all_sift_distances /= all_sift_distances.max()
        all_sift_distances = all_sift_distances
        if self.ae_features == False:
            all_sift_distances[:] = 1

        self.class_gaussian = self.early_gaussians
        self.plot_roc_curve(all_data,all_labels,all_sift_distances)
        
        neg_samples_count = float(np.sum(all_labels == self.class_num))
        pos_samples_count = float(np.sum(all_labels < self.class_num))
        if self.log_tsne:
            x_te_proj_pca = TSNE(n_components=2, perplexity=30, learning_rate=200).fit_transform(all_data)
            x_te_proj_df = pd.DataFrame(x_te_proj_pca[:, :2], columns=['Proj1', 'Proj2'])
            x_te_proj_df['label'] = all_labels
            fig = plt.figure()
            ax = sns.scatterplot('Proj1', 'Proj2', data=x_te_proj_df,
                    palette='tab20',
                    hue='label',
                    linewidth=0,
                    alpha=0.6,
                    s=7)
            box = ax.get_position()
            ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
            ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            self.trainer.logger.experiment.log_image('T-SNE With Unknown', fig, description=f'Final TEST')
        

    

    def validation_epoch_end(self, validation_step_outputs):
        all_data = torch.vstack([x[0] for x in validation_step_outputs]).cpu().numpy()
        all_labels = torch.hstack([x[1] for x in validation_step_outputs]).cpu().numpy()
        neg_samples_count = float(np.sum(all_labels == self.class_num))
        pos_samples_count = float(np.sum(all_labels < self.class_num))
        if self.log_tsne:
            x_te_proj_pca = TSNE(n_components=2, perplexity=30, learning_rate=200).fit_transform(all_data)

            x_te_proj_df = pd.DataFrame(x_te_proj_pca[:, :2], columns=['Proj1', 'Proj2'])

            x_te_proj_df['label'] = all_labels
            fig = plt.figure()
            ax = sns.scatterplot('Proj1', 'Proj2', data=x_te_proj_df,
                    palette='tab20',
                    hue='label',
                    linewidth=0,
                    alpha=0.6,
                    s=7)
            box = ax.get_position()
            ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
            ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            # self.trainer.logger.experiment.add_figure('pca', fig, global_step=self.trainer.current_epoch)
            self.trainer.logger.experiment.log_image('T-SNE', fig, description=f'epoch: {self.trainer.current_epoch}')
        if self.trainer.current_epoch >= 1:
            all_data = torch.tensor(all_data,device=self.device)
            all_labels = torch.tensor(all_labels,device=self.device)
            labels = self.classify_data(all_data)
            true_labels = labels == all_labels
            f1score = f1_score(all_labels.detach().cpu().numpy(),labels.detach().cpu().numpy(),average='macro')
            self.log(f"f1_score", f1score )
            pos_labels = float(torch.sum(true_labels[all_labels < self.class_num]))
            neg_labels = float(torch.sum(true_labels[all_labels == self.class_num]))
            
            all_data,all_labels = self.filter_samples(all_data,all_labels)
            neg_samples_count_survivers = float(torch.sum(all_labels == self.class_num))
            pos_samples_count_survivers = float(torch.sum(all_labels < self.class_num))
            self.log(f"val_survivers_neg", neg_samples_count_survivers/neg_samples_count)
            self.log(f"val_survivers_pos", pos_samples_count_survivers/pos_samples_count)
            self.log(f"accuarcy_pos", pos_labels/pos_samples_count)
            self.log(f"accuarcy_neg", neg_labels/neg_samples_count)
            print("Acc:",pos_labels/pos_samples_count,"F1",f1score) 
        else:
            self.log(f"f1_score", 0)

        choosen_index = np.random.randint(len(validation_step_outputs))
        img_batch = validation_step_outputs[choosen_index]
        choosen_index = np.random.randint(len(img_batch[1]))
        x_hat = img_batch[2][choosen_index]
        x = img_batch[3][choosen_index]
        label =  img_batch[1][choosen_index]
        self.log_images(x,x_hat,label)
          

    def log_images(self,x,x_hat,label):
        ndarr = x_hat.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
        img = Image.fromarray(np.squeeze(ndarr))
        self.trainer.logger.experiment.log_image(
            'recon_samples',
            img,
            description=' Recon Image'.format(label))
        ndarr = x.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
        img = Image.fromarray(np.squeeze(ndarr))
        self.trainer.logger.experiment.log_image(
            'true_samples',
            img,
            description=f' True Image:{label}')


    def initialize_gaussians(self):
        self.class_counts = torch.zeros(self.class_num,device=self.device)
        self.class_temp_means = torch.zeros((self.class_num,self.latent_dim),device=self.device)
        self.class_temp_cov = torch.zeros(((self.class_num,self.latent_dim,self.latent_dim)),device=self.device)
        self.class_means = torch.zeros((self.class_num,self.latent_dim),device=self.device)
        self.class_cov = torch.zeros(((self.class_num,self.latent_dim,self.latent_dim)),device=self.device)
        self.class_bounds = torch.zeros(self.class_num,device=self.device)

        ## Earlier resnet layers
        self.layer_size = self.enc_out_dim + self.latent_dim
        if self.enc_type == 'densenet':
            self.layer_size += 600
        elif self.enc_type == 'vgg':
            self.layer_size += 640
        elif self.enc_type == 'resnet18':
            self.layer_size += 448
        elif self.enc_type == 'wresnet':
            self.layer_size += 1120
        else:
            self.layer_size += 300
        self.early_temp_means = torch.zeros((self.class_num,self.layer_size),device=self.device)
        self.early_temp_cov = torch.zeros(((self.class_num,self.layer_size,self.layer_size)),device=self.device)
        self.early_means = torch.zeros((self.class_num,self.layer_size),device=self.device)
        self.early_cov = torch.zeros(((self.class_num,self.layer_size,self.layer_size)),device=self.device)

    def configure_optimizers(self):
        #Little hack to get it after device was configured (which is not in init)
        # optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        #params_list = list(self.encoder.params())+list(self.fc_mu.params())+list(self.fc_var.params)        
        finetune_params = list(list(self.fc_mu.parameters())+list(self.fc_var.parameters()))
        params = finetune_params if self.finetune and self.trainer.current_epoch < 20 else self.parameters()
        # print(len(params))
        if self.finetune and self.trainer.current_epoch < 20:
            lr = self.finetune_lr
        # elif self.trainer.current_epoch < self.margin_max_distance and not self.finetune:
        #     lr = self.finetune_lr / 10
        else:
            lr=self.lr
        if self.opt == 'sgd':
            print('SGD')
            optimizer = torch.optim.SGD(params, lr=lr,momentum=0.9, dampening=0, weight_decay=1e-4,nesterov=True)
        else:
            print('ADAM')
            optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        lr_scheduler = StepLR(optimizer, step_size=self.step_size, gamma=self.step_gamma)
        # scheduler = CyclicLR(optimizer, base_lr=self.step_gamma, max_lr=self.lr)
        # lr_scheduler = {
        #     'scheduler': scheduler,
        #     'interval': 'step', # or 'epoch'
        #     'frequency': 1
        # }
        # optimizer = torch.optim.Adamax(self.parameters(), lr=self.lr)
        # optimizer = torch.optim.RMSprop(self.parameters(), lr=self.lr)
        if self.trainer.current_epoch < 20:
            all_classes = self.class_num
            self.class_counts = torch.zeros(all_classes,device=self.device)
            self.class_temp_means = torch.zeros((all_classes,self.latent_dim),device=self.device)
            self.class_temp_cov = torch.zeros(((all_classes,self.latent_dim,self.latent_dim)),device=self.device)
            self.class_means = torch.zeros((all_classes,self.latent_dim),device=self.device)
            self.class_cov = torch.zeros(((all_classes,self.latent_dim,self.latent_dim)),device=self.device)
            self.class_bounds = torch.zeros(all_classes,device=self.device)  
            self.class_counts.requires_grad = False
            self.class_temp_cov.requires_grad = False
            self.class_cov.requires_grad = False
            self.class_temp_means.requires_grad = False
            self.class_means.requires_grad = False    
            self.precentiles = [[] for i in range(all_classes)]
        return [optimizer], [lr_scheduler]

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)

        parser.add_argument("--enc_type", type=str, default='resnet18', help="resnet18/resnet50/mnist")
        parser.add_argument("--first_conv", action='store_true')
        parser.add_argument("--maxpool1", action='store_true')
        parser.add_argument("--lr", type=float, default=1e-4)

        parser.add_argument(
            "--enc_out_dim", type=int, default=512,
            help="512 for resnet18, 2048 for bigger resnets, adjust for wider resnets"
        )
        parser.add_argument("--kl_coeff", type=float, default=0.1)
        parser.add_argument("--latent_dim", type=int, default=256)

        parser.add_argument("--batch_size", type=int, default=256)
        parser.add_argument("--num_workers", type=int, default=8)
        parser.add_argument("--data_dir", type=str, default=".")
        parser.add_argument("--generation_epoch", type=int,default=20)
        parser.add_argument("--class_num", type=int, default=10)
        parser.add_argument("--lower_bound", type=float, default = 0.05)
        parser.add_argument("--step_size", type=int, default=100)
        parser.add_argument("--step_gamma", type=float, default = 0.1)
        parser.add_argument("--cov_scaling",type=float, default = 5.0)
        parser.add_argument("--log_tsne",action='store_true')
        parser.add_argument("--recon_weight",type=float, default = 0.1)
        parser.add_argument("--gen_weight",type=float,default = 0.5)
        parser.add_argument("--weights", default=None, type=str)
        parser.add_argument("--finetune",action='store_true')
        parser.add_argument("--finetune_lr",type=float, default = 0.0001)
        parser.add_argument("--opt", type=str, default="sgd")
        parser.add_argument("--ae_features",action='store_true')


        return parser



    def calculate_gaussian(self):
            '''
            This is used in testing, to run a training epoch (with no gradients) so we can calculate the gaussians.
            '''
            class_gaussians = []
            early_gaussians = []
            for i in range(self.class_num):
                self.class_means[i,:] = self.class_temp_means[i,:]/self.class_counts[i]
                self.class_cov[i,:,:] = self.class_temp_cov[i,:,:]/self.class_counts[i] - torch.matmul(self.class_means[i,:].view(1,-1).T,self.class_means[i,:].view(1,-1))  
                # self.class_cov[i,:,:] += torch.eye(self.class_cov[i,:,:].shape[0],device=self.device)*0.000002
                self.class_cov[i,:,:] = torch.eye(self.class_cov[i,:,:].shape[0],device=self.device)*10
                self.early_means[i,:] = self.early_temp_means[i,:]/self.class_counts[i]
                self.early_cov[i,:,:] = self.early_temp_cov[i,:,:]/self.class_counts[i] - torch.matmul(self.early_means[i,:].view(1,-1).T,self.early_means[i,:].view(1,-1)) 
                self.early_cov[i,:,:] = (self.early_cov[i,:,:] + torch.t(self.early_cov[i,:,:]))/2
                # self.early_cov[i,:,:] += torch.eye(self.early_cov[i,:,:].shape[0],device=self.device)*0.000001
                # self.early_cov[i,:,:] += torch.eye(self.early_cov[i,:,:].shape[0],device=self.device)*0.0001
                try:
                    dist = torch.distributions.multivariate_normal.MultivariateNormal(self.class_means[i,:],self.class_cov[i,:,:])
                except:
                    dist = self.class_gaussian[i]            
                class_gaussians.append(dist)
            increases = 0
            for i in range(self.class_num):
                class_increase = increases
                cov_copy = torch.clone(self.early_cov[i,:,:])
                cov_copy += torch.eye(self.early_cov[i,:,:].shape[0],device=self.device)*0.000001*increases
                while True:
                    try:
                        # torch.cholesky(cov_copy)
                        torch.distributions.multivariate_normal.MultivariateNormal(self.early_means[i,:],cov_copy)
                        break
                    except:
                        cov_copy += torch.eye(self.early_cov[i,:,:].shape[0],device=self.device)*0.000001
                        class_increase += 1
                if class_increase > increases:
                    increases = class_increase
            increases+=2
            print("Increases:", increases)
            for i in range(self.class_num):
                self.early_cov[i,:,:] += torch.eye(self.early_cov[i,:,:].shape[0],device=self.device)*0.000001*increases 
                early_dist = torch.distributions.multivariate_normal.MultivariateNormal(self.early_means[i,:],self.early_cov[i,:,:])
                early_gaussians.append(early_dist)   
            
            early_gaussian_mean = self.early_means.sum(axis=0) / self.class_counts.sum()
            early_gaussian_cov = self.early_cov.sum(axis=0) / self.class_counts.sum() - torch.matmul(early_gaussian_mean.view(1,-1).T,early_gaussian_mean.view(1,-1)) 
            early_gaussian_cov = torch.eye(early_gaussian_cov.shape[0],device=self.device)*1
            early_gaussian = torch.distributions.multivariate_normal.MultivariateNormal(early_gaussian_mean, early_gaussian_cov) 
            # early_gaussians.append(early_gaussian)  
            self.unfreeze()     
            self.class_gaussian = class_gaussians
            self.early_gaussians = early_gaussians
            self.ae = ae_dict[self.known_dataset][0]()
            self.ae = self.ae.load_from_checkpoint(ae_dict[self.known_dataset][1])
