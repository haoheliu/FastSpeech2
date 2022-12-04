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
        
def listdir_nohidden(path):
    for f in os.listdir(path):
        if not f.startswith('.'):
            yield f
            
def get_restore_step(path):
    checkpoints = os.listdir(path)
    steps = [int(x.split(".ckpt")[0].split("step=")[1]) for x in checkpoints]
    return max(steps)

def main(args, configs):    

    preprocess_config, model_config, train_config, autoencoder_config = configs
    configuration = {**preprocess_config, **model_config, **train_config, **autoencoder_config} 

    preprocess_config, model_config, train_config, autoencoder_config = configs
        
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
            
        batch_size = autoencoder_config["data"]["batchsize"]
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=12,
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
            
        batch_size = autoencoder_config["data"]["batchsize"]
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=12,
            pin_memory=True
        )
        
    print("The length of the dataset is %s, the length of the dataloader is %s, the batchsize is %s" % (len(dataset), len(loader),batch_size))

    # Get dataset
    val_dataset = AudioSetDataset(
        preprocess_config, train_config, train=False
    )
    
    batch_size = autoencoder_config["data"]["batchsize"]
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=True,
    )

    model = AutoencoderKL(
                        model_config, preprocess_config, autoencoder_config,
                            autoencoder_config["model"]["params"]["ddconfig"],
                          autoencoder_config["model"]["params"]["lossconfig"],
                          embed_dim=autoencoder_config["model"]["params"]["embed_dim"],
                          learning_rate = autoencoder_config["model"]["base_learning_rate"],
                        )
    
    autoencoder_config["id"]["version"] = "%s_%s_%s_%s_%s" % (autoencoder_config["id"]["version"], 
                                  autoencoder_config["id"]["name"], 
                               autoencoder_config["model"]["params"]["embed_dim"],
                               autoencoder_config["model"]["params"]["ddconfig"]["ch"],
                               autoencoder_config["model"]["base_learning_rate"])
    
    wandb_logger = WandbLogger(version=autoencoder_config["id"]["version"], 
                               project="audioverse",config=configuration, name=autoencoder_config["id"]["name"], save_dir="log") 
    
    checkpoint_callback = ModelCheckpoint(
        monitor='train/total_loss',
        filename='checkpoint-{step:02d}',
        every_n_train_steps=10000,
        save_top_k=5
    )

    checkpoint_path = os.path.join(autoencoder_config["id"]["root"],"log",autoencoder_config["id"]["version"])
    os.makedirs(checkpoint_path, exist_ok=True)
    
    if(len(os.listdir(checkpoint_path)) > 1):
        resume_from_checkpoint=os.path.join(checkpoint_path, get_restore_step(checkpoint_path))
        print("Resume from checkpoint", resume_from_checkpoint)
    else:
        print("Train from scratch")
        resume_from_checkpoint = None
    
    trainer = Trainer(resume_from_checkpoint=resume_from_checkpoint,
                        accelerator="gpu", 
                      devices=torch.cuda.device_count(), 
                      logger=wandb_logger, 
                      callbacks=[checkpoint_callback],
                    #   val_check_interval=5000,
                    #   limit_train_batches=5,
                      limit_val_batches=500,
                      )

    trainer.fit(model, loader, val_loader)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c", "--autoencoder_config", type=str, required=True, help="path to autoencoder config folder"
    )    

    args = parser.parse_args()
    
    config_root = args.autoencoder_config
    
    preprocess_config = os.path.join(config_root,"preprocess.yaml")
    model_config = os.path.join(config_root,"model.yaml")
    train_config = os.path.join(config_root,"train.yaml")
    autoencoder_config = os.path.join(config_root, "config.yaml")
    
    preprocess_config = yaml.load(open(preprocess_config, "r"), Loader=yaml.FullLoader)
    model_config = yaml.load(open(model_config, "r"), Loader=yaml.FullLoader)
    train_config = yaml.load(open(train_config, "r"), Loader=yaml.FullLoader)
    autoencoder_config = yaml.load(open(autoencoder_config, "r"), Loader=yaml.FullLoader)
    
    configs = (preprocess_config, model_config, train_config, autoencoder_config)
    
    main(args, configs)