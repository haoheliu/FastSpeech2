import sys
sys.path.append("/mnt/fast/nobackup/users/hl01486/projects/general_audio_generation/conditional_transfer/FastSpeech2")

import os
import numpy as np

import argparse
import yaml
import torch

from dataset import Dataset as AudioSetDataset

from torch.utils.data import WeightedRandomSampler
from torch.utils.data import DataLoader, Dataset
# from _utils.sampler import DistributedSamplerWrapper
# from torch.utils.data.distributed import DistributedSampler
from pytorch_lightning import Trainer, seed_everything
from ldm.models.autoencoder import AutoencoderKL
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import wandb
        
def get_restore_step(path):
    checkpoints = os.listdir(path)
    steps = [int(x.split(".")[0]) for x in checkpoints]
    return max(steps)

def main(args, configs):    

    preprocess_config, model_config, train_config, autoencoder_config = configs
    configuration = {**preprocess_config, **model_config, **train_config, **autoencoder_config} 
    wandb.init(
    project="audioverse",
    name="Encoder_Decoder",
    notes="",
    tags=["autoencoderkl"],
    config=configuration,
    )

    preprocess_config, model_config, train_config, autoencoder_config = configs
    ckpt_path = os.path.join(train_config["path"]["ckpt_path"])
    
    if(os.path.exists(ckpt_path) and len(os.listdir(ckpt_path)) > 0):
        args.restore_step = get_restore_step(ckpt_path)
        
    # Get dataset
    if(train_config["augmentation"]["balanced_sampling"]):
        print('balanced sampler is being used')
        samples_weight = np.loadtxt(preprocess_config["path"]["train_data"][:-5] + '_weight.csv', delimiter=',')
        sampler = WeightedRandomSampler(samples_weight, len(samples_weight), replacement=True)
        
        dataset = AudioSetDataset(
            preprocess_config, train_config, train=True
        )
        
        # if(n_gpus>1):
        #     sampler = DistributedSamplerWrapper(sampler, num_replicas=n_gpus, rank=rank, shuffle=True)
            
        batch_size = train_config["optimizer"]["batch_size"]
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=16,
            pin_memory=True
        )
    else:
        print('balanced sampler is not used')
        dataset = AudioSetDataset(
            preprocess_config, train_config, train=True
        )
        # if(n_gpus > 1):
        #     sampler = DistributedSampler(dataset, num_replicas=n_gpus, rank=rank, shuffle=True)
        # else:
        #     sampler = None
            
        batch_size = train_config["optimizer"]["batch_size"]
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=16,
            pin_memory=True
        )
    print("The length of the dataset is %s, the length of the dataloader is %s, the batchsize is %s" % (len(dataset), len(loader),batch_size))

    # Get dataset
    val_dataset = AudioSetDataset(
        preprocess_config, train_config, train=False
    )
    
    batch_size = train_config["optimizer"]["batch_size"]
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=True,
    )

    model = AutoencoderKL(preprocess_config,
                          autoencoder_config["model"]["params"]["ddconfig"],
                          autoencoder_config["model"]["params"]["lossconfig"],
                          embed_dim=autoencoder_config["model"]["params"]["embed_dim"],
                          learning_rate = autoencoder_config["model"]["base_learning_rate"],
                        )
    
    wandb_logger = WandbLogger() 
    
    checkpoint_callback = ModelCheckpoint(
         monitor='val/rec_loss',
         filename='checkpoint-{epoch:02d}-{val_loss:.2f}',
         every_n_epochs=1,
         save_top_k=5
    )

    trainer = Trainer(accelerator="gpu", 
                      devices=torch.cuda.device_count(), 
                      logger=wandb_logger, 
                      callbacks=[checkpoint_callback],
                      limit_val_batches=200)

    trainer.fit(model, loader, val_loader)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--restore_step", type=int, default=0)
    parser.add_argument(
        "-p",
        "--preprocess_config",
        type=str,
        required=True,
        help="path to preprocess.yaml",
    )
    parser.add_argument(
        "-m", "--model_config", type=str, required=True, help="path to model.yaml"
    )
    parser.add_argument(
        "-t", "--train_config", type=str, required=True, help="path to train.yaml"
    )    
    
    args = parser.parse_args()
    
    preprocess_config = yaml.load(
        open(args.preprocess_config, "r"), Loader=yaml.FullLoader
    )
    model_config = yaml.load(open(args.model_config, "r"), Loader=yaml.FullLoader)
    train_config = yaml.load(open(args.train_config, "r"), Loader=yaml.FullLoader)
    
    autoencoder_config = yaml.load(open("/mnt/fast/nobackup/users/hl01486/projects/general_audio_generation/conditional_transfer/FastSpeech2/encoder_decoder/config_model/kl-f8/config.yaml", "r"), Loader=yaml.FullLoader)
    
    configs = (preprocess_config, model_config, train_config, autoencoder_config)
    
    main(args, configs)