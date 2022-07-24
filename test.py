import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), "lib"))
from argparse import ArgumentParser
import pytorch_lightning as pl
import matplotlib
matplotlib.use('Agg')
from models.vae import GMM_VAE_Contrastive,UpdateMargin,InitializeGaussians

from data.mnist_module import MNISTDataModule
from data.open_set_data_loader import KnownUnknownDataModule

from pytorch_lightning.loggers.neptune import NeptuneLogger
from pytorch_lightning.callbacks import LearningRateMonitor,ModelCheckpoint

def cli_main(args=None):  

    pl.seed_everything()

    parser = ArgumentParser()
    parser.add_argument("--checkpoint_path", default=None, type=str)
    parser.add_argument("--margin_max_distance", type=int, default=32)
    parser.add_argument("--known_dataset", default="mnist", type=str)
    parser.add_argument("--unknown_dataset", default="omniglot", type=str)
    parser.add_argument("--data_transforms", default="mnist", type=str)
    parser.add_argument("--input_height", type=int, default=28)
    parser.add_argument("--margin_jumps", type=float, default=1.0)
    parser.add_argument("--margins_epoch", type=int, default=10)
    parser.add_argument("--offline",action='store_true')
    parser.add_argument("--batch_classes", type=int, default=10)
    parser.add_argument("--sample_count", type=int, default=40)
    script_args, _ = parser.parse_known_args(args)

    parser = GMM_VAE_Contrastive.add_model_specific_args(parser)




    
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args(args)

    dm = KnownUnknownDataModule.from_argparse_args(args,is_testing=True)
    # dm = MNISTDataModule.from_argparse_args(args)
    # args.input_height = dm.size()[-1]
    print(args.input_height)

    if args.max_steps == -1:
        args.max_steps = None
        

    model = GMM_VAE_Contrastive.load_from_checkpoint(**vars(args))

    neptune_logger = NeptuneLogger(
        api_key=None,
        # project_name="",
        # experiment_name="",
        params=vars(args),
        offline_mode=script_args.offline,
        tags=[f'test',f'known:{script_args.known_dataset}',f'unknown:{script_args.unknown_dataset}',f'backbone:{args.enc_type}'])

    lr_logger = LearningRateMonitor(logging_interval='epoch')



    trainer = pl.Trainer.from_argparse_args(args,callbacks = [InitializeGaussians()],logger=neptune_logger)
    # trainer = pl.Trainer.from_argparse_args(args)
    trainer.test(model, dm.train_dataloader()) # First iteration will calculate Gaussians, second will do the test.
    trainer.test(model, dm.test_dataloader())
    return dm, model, trainer


if __name__ == "__main__":
    dm, model, trainer = cli_main()