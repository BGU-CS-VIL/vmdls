from argparse import ArgumentParser

import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.nn import functional as F
from torchvision import transforms
from pl_bolts.transforms.dataset_normalizations import cifar10_normalization

from pl_bolts.models.autoencoders.components import resnet18_encoder, resnet18_decoder
from pl_bolts.models.autoencoders.components import resnet50_encoder, resnet50_decoder
from models.decoders_encoders import MnistConvEncoder, MnistConvDecoder


dt = {'mnist': {
        'train':transforms.Compose([
            transforms.RandomRotation(10, fill=(0,)),
            transforms.RandomResizedCrop(28, scale=(0.95,1)),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.5,), (0.5,))]),
        'val':transforms.Compose([            
            transforms.ToTensor(),
            transforms.Normalize(
               (0.5,), (0.5,))]),
        'test':transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                (0.5,), (0.5,))])}}


class MNIST_AE(pl.LightningModule):

    pretrained_urls = {
        'cifar10-resnet18':
            'https://pl-bolts-weights.s3.us-east-2.amazonaws.com/ae/ae-cifar10/checkpoints/epoch%3D96.ckpt'
    }

    def __init__(
        self,
        input_height,
        enc_type='mnist',
        first_conv=False,
        maxpool1=False,
        enc_out_dim=500,
        kl_coeff=0.1,
        latent_dim=32,
        lr=1e-4,
        **kwargs
    ):
        """
        Standard AE

        Model is available pretrained on different datasets:

        Example::

            # not pretrained
            ae = AE()

            # pretrained on cifar10
            ae = AE.from_pretrained('cifar10-resnet18')

        Args:

            input_height: height of the images
            enc_type: option between resnet18 or resnet50
            first_conv: use standard kernel_size 7, stride 2 at start or
                replace it with kernel_size 3, stride 1 conv
            maxpool1: use standard maxpool to reduce spatial dim of feat by a factor of 2
            enc_out_dim: set according to the out_channel count of
                encoder used (512 for resnet18, 2048 for resnet50)
            latent_dim: dim of latent space
            lr: learning rate for Adam
        """

        super(MNIST_AE, self).__init__()

        self.save_hyperparameters()

        self.lr = lr
        self.enc_out_dim = enc_out_dim
        self.latent_dim = latent_dim
        self.input_height = input_height

        valid_encoders = {
            'resnet18': {'enc': resnet18_encoder, 'dec': resnet18_decoder},
            'resnet50': {'enc': resnet50_encoder, 'dec': resnet50_decoder},
            'mnist':{'enc': MnistConvEncoder, 'dec': MnistConvDecoder},
        }

        if enc_type not in valid_encoders:
            self.encoder = resnet18_encoder(first_conv, maxpool1)
            self.decoder = resnet18_decoder(self.latent_dim, self.input_height, first_conv, maxpool1)
        else:
            self.encoder = valid_encoders[enc_type]['enc'](first_conv, maxpool1)
            self.decoder = valid_encoders[enc_type]['dec'](self.latent_dim, self.input_height, first_conv, maxpool1)

        self.fc = nn.Linear(self.enc_out_dim, self.latent_dim)


    def forward(self, x):
        feats = self.encoder(x)
        z = self.fc(feats)
        x_hat = self.decoder(z)
        return x_hat

    def step(self, batch, batch_idx):
        x, y = batch

        feats = self.encoder(x)
        z = self.fc(feats)
        x_hat = self.decoder(z)

        loss = F.mse_loss(x_hat, x, reduction='mean')

        return loss, {"loss": loss}

    def training_step(self, batch, batch_idx):
        loss, logs = self.step(batch, batch_idx)
        self.log_dict(
            {f"train_{k}": v for k, v in logs.items()}, on_step=True, on_epoch=False
        )
        return loss

    def validation_step(self, batch, batch_idx):
        loss, logs = self.step(batch, batch_idx)
        self.log_dict({f"val_{k}": v for k, v in logs.items()})
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)

        parser.add_argument("--enc_type", type=str, default='resnet18', help="resnet18/resnet50")
        parser.add_argument("--first_conv", action='store_true')
        parser.add_argument("--maxpool1", action='store_true')
        parser.add_argument("--lr", type=float, default=1e-4)

        parser.add_argument(
            "--enc_out_dim", type=int, default=500,
            help="500"
        )
        parser.add_argument("--latent_dim", type=int, default=32)

        parser.add_argument("--batch_size", type=int, default=256)
        parser.add_argument("--num_workers", type=int, default=8)
        parser.add_argument("--data_dir", type=str, default=".")

        return parser


def cli_main(args=None):
    from pl_bolts.datamodules import CIFAR10DataModule, ImagenetDataModule, STL10DataModule, MNISTDataModule

    parser = ArgumentParser()
    parser.add_argument("--dataset", default="mnist", type=str, choices=["mnist","cifar10", "stl10", "imagenet"])
    script_args, _ = parser.parse_known_args(args)

    if script_args.dataset == "cifar10":
        dm_cls = CIFAR10DataModule
    elif script_args.dataset == "stl10":
        dm_cls = STL10DataModule
    elif script_args.dataset == "mnist":
        dm_cls = MNISTDataModule
    elif script_args.dataset == "imagenet":
        dm_cls = ImagenetDataModule
    else:
        raise ValueError(f"undefined dataset {script_args.dataset}")

    parser = MNIST_AE.add_model_specific_args(parser)
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args(args)

    dm = dm_cls.from_argparse_args(args)
    args.input_height = dm.size()[-1]

    if args.max_steps == -1:
        args.max_steps = None

    model = MNIST_AE(**vars(args))

    trainer = pl.Trainer.from_argparse_args(args)
    trainer.fit(model, dm)
    return dm, model, trainer


if __name__ == "__main__":
    dm, model, trainer = cli_main()
