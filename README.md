## Variational- and Metric-based Deep Latent Space for Out-of-Distribution Detection
[Paper](https://openreview.net/pdf?id=ScLeuUUi9gq)
### Requirments:

* python = 3.8
* torch = 1.7
* torchvision >= 0.8.2
* pytorch-lightning  = 1.1.2
* pytorch-lightning-bolts = 0.2.5
* pytorch-metric-learning = 0.9.95
* neptune-client = 0.9.1
* neptune-contrib = 0.25.0
* pandas = 1.2.0
* pillow = 7.2.0
* tqdm
* scikit-learn
* kornia = 0.5.1

Optional:
* tsnecuda 3.0.0 (For faster tsne plots)


You can use the attached yaml file for easy install.

### Whats included?

All the code required to train/test and reproduce the results.
Due to size limitations, we have included all the checkpoints and weights in the following anonymized google drive:
[Google Drive](https://drive.google.com/drive/folders/10lGq9LNyaWhFaN0eCGQd_sMaTob87ZaL?usp=sharing)

### Setup

#### AutoEncoder Weights

Download the files `ae_mnist.ckpt` and `ae_cifar100.ckpt`, place both of them under the `models` directory. Those are the weights for the auto encoders (which are used during the reconstruction weighting).
Alternatively, you can run the files `train_resnet_ae.py` and `mnist_ae.py` in order to train the autoencoders for mnsit and CIFAR-100,
in that case you will need to update the paths in the file `models/vae.py`.
CIFAR-10 uses pretrained pytorch-lighting AE.

#### Datasets
Go to [ODIN Github repository](https://github.com/facebookresearch/odin), download and extract the datasets Tiny-ImageNet Crop, Tiny-ImageNet Resize, LSUN Crop, LSUN resize, and place them under `data/openset/Imagenet`,
`data/openset/Imagenet_resize`,`data/openset/LSUN`,`data/openset/LSUN_resize` respectivley.

Download the [Omniglot test set](https://github.com/brendenlake/omniglot/raw/master/python/images_evaluation.zip) and place it under `data/omniglot-py/images_evaluation/`.

The rest of the datasets will be downloaded during training/testing.

#### Finetuning weights
For the Cifar100 - Download `checkpoint_100.pth.tar` from the google drive, or as an alternative you can download from [here](https://drive.google.com/drive/folders/1HOG05gDorYBxaKXGrF9nXnvYrVogsZaw), which is the version we have finetuned.
For the WideResnet - Download `finetuning_wideresnet.tar` from the google drive.

#### Checkpoints
We have included the checkpoints for the experiments, download the relevant checkpoints and use them for testing as described below.
Place the checkpoints in the main directory.

#### Neptune
Neptune is used for all the logging done. While it is not mandatory, more information is passes to the logger.
If you wish to use neptune, update the files `main.py` and `test.py` with your neptune details, specificity,
you should update the neptune logger project, experiment and API key (optional, an enviroment variable can be used instead).
To disable neptune logging, add `--offline` argument, note that you should still have it installed, and some logs are only saved to neptune.

### Testing:

The testing is done via the file `test.py`, to reproduce the results in the paper use the following arguments:

#### CIFAR-10

DenseNet:
`python test.py --enc_type densenet --latent_dim 32 --enc_out_dim 342 --max_epochs 500 --gpus "0,"  --generation_epoch 5 --lower_bound 0.05 --data_dir ./data --step_size 200 --margin_max_distance 64 --known_dataset cifar_known --unknown_dataset imagenet-resize --data_transforms cifar_test --input_height 32 --class_num 10 --kl_coeff 0.1 --lr 0.001 --step_size 100 --margin_jump 1.0 --margins_epoch 30 --checkpoint_path cifar10_densenet.ckpt --batch_classes 1 --sample_count 4996 --batch_size 2500 --ae_features`

ResNet:
`python test.py --enc_type resnet18 --latent_dim 32 --enc_out_dim 512 --max_epochs 500 --gpus "0," --generation_epoch 1 --lower_bound 0.98 --data_dir ./data --step_size 100 --margin_max_distance 32 --known_dataset cifar_known --unknown_dataset lsun-crop --data_transforms cifar_test --input_height 32 --class_num 10 --kl_coeff 0.1 --lr 0.1 --margin_jump 1.0 --margins_epoch 10 --batch_classes 1 --sample_count 4996 --batch_size 2500 --checkpoint_path cifar10_resnet.ckpt --ae_features`

VGG:
`python test.py --enc_type vgg --latent_dim 32 --enc_out_dim 1024 --max_epochs 500 --gpus "0," --generation_epoch 0 --lower_bound 0.05 --data_dir ./data --step_size 100 --margin_max_distance 32 --known_dataset cifar_known --unknown_dataset imagenet-crop --data_transforms cifar_test --input_height 32 --class_num 10 --kl_coeff 0.1 --lr 0.001 --margin_jump 1.0 --margins_epoch 10 --batch_classes 1 --sample_count 2500  --batch_size 2500 --checkpoint_path cifar10_vgg.ckpt --ae_features`

WideResNet:
`python test.py --enc_type wresnet --latent_dim 32 --enc_out_dim 640 --max_epochs 500 --gpus "0,"  --generation_epoch 5 --lower_bound 0.05 --data_dir ./data --step_size 200 --margin_max_distance 32 --known_dataset cifar_known --unknown_dataset imagenet-crop --data_transforms cifar_test --input_height 32 --class_num 10 --kl_coeff 0.1 --lr 0.001 --step_size 100 --margin_jump 1.0 --margins_epoch 30 --checkpoint_path cifar10_wresnet.ckpt --batch_classes 2 --sample_count 2500 --batch_size 2000`

#### CIFAR-100
`python test.py --enc_type densenet --latent_dim 64 --enc_out_dim 342 --max_epochs 500 --gpus "0," --generation_epoch 5 --lower_bound 0.05 --data_dir ./data --step_size 100 --margin_max_distance 64 --known_dataset cifar100_known --unknown_dataset lsun-resize --data_transforms cifar_test --input_height 32 --class_num 100 --kl_coeff 0.1 --lr 0.01 --margin_jump 1.0 --margins_epoch 10 --batch_classes 10 --sample_count 450 --batch_size 500 --checkpoint_path cifar100_densenet.ckpt --ae_features`

#### MNIST
` python test.py --enc_type mnist-conv --latent_dim 32 --enc_out_dim 500 --max_epochs 500 --gpus "0," --generation_epoch 151 --lower_bound 0.05 --data_dir ./data --step_size 75 --margin_max_distance 32 --known_dataset mnist --unknown_dataset omniglot --data_transforms blob --input_height 28 --class_num 10 --kl_coeff 0.1 --lr 0.1 --margin_jump 1.0 --margins_epoch 10 --batch_classes 1 --sample_count 4996 --batch_size 2500 --checkpoint_path cnn_mnist.ckpt --ae_features`

#### Relevant Arguments
* --ae_features use/disable the reconstruction
* --unknown_dataset valid values are `imagenet-crop`,`imagenet-resize`,`lsun-crop`,`lsun-resize`,`omniglot`,`mnist_noise`,`noise`.
* --thresholding This is used for the F1-score experiments, and outputs the thresholds to a table (neptune required), we then use the threshold at 95% tpr, and run it again without the argument, which will output the f1 per threshold table, from where the score can be taken.

Note that disabling the reconstruction might require to alter the F1 thresholding in the code, as the values are in a different scale, for efficiency in the testing it only scans values are are relevant when the reconstruction is enabled.
To alter the values modify the function `log_test_statistics` in `models/vae.py`.

### Training

The training is done via the file `main.py`, with the following possible arguments:

* --enc_type The backbone type, valid values are `vgg/resnet18/resnet50/densenet/mnist-conv`.
* --latent_dim The latent dim size.
* --enc_out_dim the output size of the encoder, `1024/512/2048/342/500` for `vgg/resnet18/resnet50/densenet/mnist-conv`.
* --max_epochs
* --generation_epoch the epochs where we start using the KL loss, `1` is the preferred value (as the means are calculate from the previous epoch)
* --lower_bound Used to calculate F1-score mid run, does not affect results, preferred value is `0.95`.
* --data_dir the location of the datasets, `-./data` is preferred.
* --margin_max_distance `M_d` hyperparams
* --known_dataset the known dataset used for training, valid values are `mnist/cifar_known/cifar100_known`.
* --unknown_dataset unknown dataset used for mid-training F1-score, valid values are `imagenet-crop/imagenet-resize/lsun-crop/lsun-resize/omniglot/mnist_noise/noise/cifar_unknown/cifar100_unknown`, the latter two may only be used with `cifar_known/cifar100_known` respectively, with `class-num` lower then the actual number of classes (used to train either cifar10 or cifar100 with part of the classes as known, and part unknown).
* --data_transforms valid values for training are `mnist/cifar`.
* --input_height the size of the input, `32/28` for `cifar/mnist`.
* --class_num the number of classes, should be the number of classes in the dataset, choosing a lower number will only train on classes `[0,class_num]`.
* --lr starting learning rate
* --step_size reduce the learning rate on `step_size` epochs by `1/10`.
* --margin_jump During the warmpup we increase the `M_d` by `margin_jump`, optimal value is `1.0`.
* --margin_epochs when to start increasing `M_d`, optimal value is `0` for normal training, `20` for finetune.
* --batch_classes `B_c`.
* --sample_count `B_s`, this is the starting `B_s`, thus should be `1/5` of the values in the experiments tables (if you attempt to reproduce those), unless in finetune, which in that case should be the value from the table.
* --batch_size Validation batch size.
* --opt either `adam` or `sgd`.
* --finetune_lr If you use finetune, this is the learning rate for the first 20 epochs, for the `\mu` and `\sigma` layers. 
* --finetune if specified will use finetune
* --checkpoint_path used for resuming previous training
* --weights if you wish to load the backbone with pretrained weights.
* --offline will not upload results to neptune

For example, if you wish to reproduce the `CIFAR-10` experiment with `VGG`, you would use:
```
python main.py --enc_type vgg --latent_dim 32 --enc_out_dim 1024 --max_epochs 500 --gpus "1,"  --generation_epoch 0 --lower_bound 0.95 --data_dir ./data --margin_max_distance 32 --known_dataset cifar_known --unknown_dataset imagenet-resize --data_transforms cifar --input_height 32 --class_num 10 --lr 0.001 --step_size 100 --margin_jump 1.0 --margins_epoch 10 --batch_classes 8 --sample_count 8  --batch_size 64 --opt adam
```

For the `CIFAR-10` experiment with `ResNet18`:
```
python main.py --enc_type resnet18 --latent_dim 32 --enc_out_dim 512 --max_epochs 500 --gpus "0,"  --generation_epoch 0 --lower_bound 0.95 --data_dir ./data --margin_max_distance 32 --known_dataset cifar_known --unknown_dataset imagenet-resize --data_transforms cifar --input_height 32 --class_num 10 --kl_coeff 0.1 --lr 0.01 --step_size 100 --margin_jump 1.0 --margins_epoch 10 --batch_classes 10 --sample_count 8  --batch_size 400 --opt sgd
```

For the `CIFAR-10` experiment with `DenseNet`:
```
python main.py --enc_type densenet --latent_dim 32 --enc_out_dim 342 --max_epochs 500 --gpus "1,"  --generation_epoch 0 --lower_bound 0.95 --data_dir ./data --margin_max_distance 32 --known_dataset cifar_known --unknown_dataset imagenet-resize --data_transforms cifar --input_height 32 --class_num 10 --kl_coeff 0.1 --lr 0.1 --step_size 100 --margin_jump 1.0 --margins_epoch 30 --batch_classes 10 --sample_count 8  --batch_size 420 --opt sgd
```

Or the `MNIST` experiment (note that here you will get good results even with only 100 epochs):
```
 python main.py --enc_type mnist-conv --latent_dim 32 --enc_out_dim 500 --max_epochs 500 --gpus "0," --generation_epoch 0 --lower_bound 0.05 --data_dir ./data --step_size 100 --margin_max_distance 32 --known_dataset mnist --unknown_dataset omniglot --data_transforms mnist --input_height 28 --class_num 10 --kl_coeff 0.1 --lr 0.0001 --opt adam --margin_jump 1.0 --margins_epoch 1 --batch_classes 10 --sample_count 16 --batch_size 400
 ```

Or, as another example, if we wish to reproduce the `CIFAR-100` finetune experiment:
```
python main.py --enc_type densenet --latent_dim 64 --enc_out_dim 342 --max_epochs 200 --gpus "0," --generation_epoch 0 --lower_bound 0.05 --data_dir ./data --step_size 64 --margin_max_distance 64 --known_dataset cifar100_known 
--unknown_dataset imagenet-resize --data_transforms cifar --input_height 32 --class_num 100 --lr 0.001 --opt sgd --margin_jump 1.0 --margins_epoch 40 --batch_classes 20 --sample_count 20 --batch_size 400 --weights checkpoint_100.pth.tar --finetune_lr 0.01 --finetune
```

For the fine-tune widereset experiment:
```
python main.py --enc_type wresnet --latent_dim 32 --enc_out_dim 640 --max_epochs 500 --gpus "0,"  --generation_epoch 0 --lower_bound 0.95 --data_dir ./data --margin_max_distance 32 --known_dataset cifar_known --unknown_dataset imagenet-resize --data_transforms cifar --input_height 32 --class_num 10 --kl_coeff 0.1 --lr 0.001 --step_size 100 --margin_jump 1.0 --margins_epoch 40 --batch_classes 10 --sample_count 20  --batch_size 640 --weights finetuning_wideresnet.tar --finetune_lr 0.01 --finetune
```

In general, we have used the last checkpoint of the training for the experiments, and most statistics are uploaded to neptune. However we do print the accuracy of the model (out of the positive samples) and the F1-score. Those are not the <i>true</i> values, as midtraining we only use the ultimate layer, and not the feature ensemble.


### Citing this work
If you use this code for your work, please cite the following:

```
@inproceedings{dinari2022variational,
  title={Variational-and Metric-based Deep Latent Space for Out-of-Distribution Detection},
  author={Dinari, Or and Freifeld, Oren},
  booktitle={The 38th Conference on Uncertainty in Artificial Intelligence},
  year={2022}
}
```