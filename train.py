import argparse
import os

import torch
import yaml
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import time
import numpy as np
from torch.utils.data import WeightedRandomSampler
import random
import torch.multiprocessing as mp
from utils.sampler import DistributedSamplerWrapper
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import pandas as pd

import matplotlib.pyplot as plt
from utils.model import get_model, get_vocoder, get_param_num, get_discriminator
from utils.tools import to_device, log, synth_one_sample
from model import FastSpeech2Loss
from dataset import Dataset
from evaluate import evaluate
import math

def build_id_to_label(
    path,
):
    ret = {}
    id2num = {}
    num2label = {}
    df = pd.read_csv(path)
    for _, row in df.iterrows():
        index, mid, display_name = row["index"], row["mid"], row["display_name"]
        ret[mid] = display_name
        id2num[mid] = index
        num2label[index] = display_name
    return ret, id2num, num2label

def cleanup():
    dist.destroy_process_group()
    
# def seed_worker(worker_id):
#     worker_seed = torch.initial_seed() % 2**32
#     np.random.seed(worker_seed)
#     random.seed(worker_seed)
        
def seed_torch(seed=0):
    # print("Set seed to %s" % seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

FBANK=None
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_restore_step(path):
    checkpoints = os.listdir(path)
    steps = [int(x.split(".")[0]) for x in checkpoints]
    return max(steps)

def main(rank, n_gpus, args, configs):
    global FBANK
    
    print(f"Running DDP on rank {rank}.")
    print("Prepare training ...")
    
    if(n_gpus > 1):
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = '60000'
        # dist.init_process_group("gloo", rank=rank, world_size=n_gpus)
        dist.init_process_group(backend='nccl', init_method='env://', world_size=n_gpus, rank=rank)
        
    device = torch.device("cuda:%s" % rank if torch.cuda.is_available() else "cpu")
    
    torch.cuda.set_device(rank)
    torch.set_grad_enabled(True)
    
    torch.manual_seed(0) 
    
    g = torch.Generator()
    g.manual_seed(0)
    
    print("I am process %s, running on %s: starting (%s)" % (
            os.getpid(), os.uname()[1], time.asctime()))
    
    preprocess_config, model_config, train_config = configs
    ckpt_path = os.path.join(train_config["path"]["ckpt_path"])
    fbank_mean = preprocess_config["preprocessing"]["mel"]["mean"]
    fbank_std = preprocess_config["preprocessing"]["mel"]["std"]
    _,_,num2label = build_id_to_label(preprocess_config["path"]["class_label_index"])
    
    def normalize(x):
        return (x-fbank_mean)/fbank_std

    def denormalize(x):
        return x * fbank_std + fbank_mean
    
    if(os.path.exists(ckpt_path) and len(os.listdir(ckpt_path)) > 0):
        args.restore_step = get_restore_step(ckpt_path)
        
    # Get dataset
    if(train_config["augmentation"]["balanced_sampling"]):
        print('balanced sampler is being used')
        samples_weight = np.loadtxt(preprocess_config["path"]["train_data"][:-5] + '_weight.csv', delimiter=',')
        sampler = WeightedRandomSampler(samples_weight, len(samples_weight), replacement=True)
        dataset = Dataset(
            preprocess_config, train_config, train=True
        )
        
        if(n_gpus>1):
            sampler = DistributedSamplerWrapper(sampler, num_replicas=n_gpus, rank=rank, shuffle=True)
            
        batch_size = train_config["optimizer"]["batch_size"]
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=16,
            generator=g,
            pin_memory=True
        )
    else:
        print('balanced sampler is not used')
        dataset = Dataset(
            preprocess_config, train_config, train=True
        )
        if(n_gpus > 1):
            sampler = DistributedSampler(dataset, num_replicas=n_gpus, rank=rank, shuffle=True)
        else:
            sampler = None
            
        batch_size = train_config["optimizer"]["batch_size"]
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=0,
            # worker_init_fn=seed_worker,
            generator=g,
            pin_memory=True
        )
    print("The length of the dataset is %s, the length of the dataloader is %s, the batchsize is %s" % (len(dataset), len(loader),batch_size))

    # disc, opt_d = get_discriminator(args, configs, device, train=True)
    # Prepare model
    model, optimizer = get_model(args, model_config["model_name"], configs, device, train=True)
    print(model)
    model = model.cuda(rank)
    print("===> Woking directory:", os.getcwd())
    if(n_gpus > 1):
        model = DDP(model, device_ids=[rank])
        
    num_param = get_param_num(model)
    # Loss = FastSpeech2Loss(preprocess_config, model_config).to(device)
    print("Number of FastSpeech2 Parameters:", num_param)

    # Load vocoder 
    vocoder = get_vocoder(model_config, device, preprocess_config["preprocessing"]["mel"]["n_mel_channels"])

    # Init logger
    for p in train_config["path"].values():
        os.makedirs(p, exist_ok=True)
    train_log_path = os.path.join(train_config["path"]["log_path"], "train")
    val_log_path = os.path.join(train_config["path"]["log_path"], "val")
    os.makedirs(train_log_path, exist_ok=True)
    os.makedirs(val_log_path, exist_ok=True)
    train_logger = SummaryWriter(train_log_path)
    val_logger = SummaryWriter(val_log_path)

    # Training
    step = args.restore_step + 1
    grad_acc_step = train_config["optimizer"]["grad_acc_step"]
    grad_clip_thresh = train_config["optimizer"]["grad_clip_thresh"]
    total_step = train_config["step"]["total_step"]
    log_step = train_config["step"]["log_step"]
    save_step = train_config["step"]["save_step"]
    synth_step = train_config["step"]["synth_step"]
    val_step = train_config["step"]["val_step"]

    outer_bar = tqdm(total=total_step, desc="Training", position=0)
    outer_bar.n = args.restore_step
    outer_bar.update()

    while True:
        for batchs in tqdm(loader):
            # fbank: [4, 1000, 80], labels: [4, 309]
            fbank, labels, fnames, waveform, seg_label = batchs 
            
            # for i in range(fbank.size(0)):
            #     fb = fbank[i].numpy()
            #     seg_lb = seg_label[i].numpy()
            #     logits = np.mean(seg_lb, axis=0)
            #     index = np.argsort(logits)[::-1][:5]
            #     plt.imshow(seg_lb[:,index], aspect="auto")
            #     plt.title(index)
            #     plt.savefig("%s_label.png" % i)
            #     plt.close()
            #     plt.imshow(fb, aspect="auto")
            #     plt.savefig("%s_fb.png" % i)
            #     plt.close()

            fbank = fbank.to(device)
            # labels = labels.to(device)
            seg_label = seg_label.to(device)
            
            # if(FBANK is None):
            #     FBANK = fbank.flatten()
            # else:
            #     FBANK = torch.cat([FBANK, fbank.flatten()])
            # print(torch.mean(FBANK), torch.std(FBANK))
            
            fbank = normalize(fbank)
            # Forward
            diff_loss, _ = model(fbank, seg_label, gen=False)
            
            total_loss = diff_loss / grad_acc_step
            total_loss.backward()
            
            # Clipping gradients to avoid gradient explosion
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip_thresh)
            optimizer.step_and_update_lr()
            optimizer.zero_grad()

            if step % log_step == 0 or step < 10:
                lr = optimizer.get_lr()
                message1 = "Step {}/{}, ".format(step, total_step)
                message2 = "Diff Loss: %.4f, lr %s" % (diff_loss, lr)

                with open(os.path.join(train_log_path, "log.txt"), "a") as f:
                    f.write(message1 + message2 + "\n")

                outer_bar.write(message1 + message2)
                log(train_logger, step)

            if(rank==0):
                if step % synth_step == 0:
                    with torch.no_grad():
                        model.eval()
                        diff_loss, generated = model(fbank, seg_label, gen=True)
                    model.train()
                    for i in range(fbank.size(0)):
                        label = [int(x) for x in torch.where(labels[i] == 1)[0]]
                        label = [num2label[x] for x in label]
                        
                        fig, wav_reconstruction, wav_prediction = synth_one_sample(
                            denormalize(fbank[i]),
                            denormalize(generated[i]),
                            label,
                            vocoder,
                            model_config,
                            preprocess_config,
                        )
                        log(
                            train_logger,
                            fig=fig,
                            tag="Training/step_{}_{}_{}".format(step, label, i),
                        )
                        sampling_rate = preprocess_config["preprocessing"]["audio"][
                            "sampling_rate"
                        ]
                        log(
                            train_logger,
                            audio=waveform[i],
                            sampling_rate=sampling_rate,
                            tag="Training/step_{}_{}_{}_original".format(step, label, i),
                        )
                        log(
                            train_logger,
                            audio=wav_reconstruction,
                            sampling_rate=sampling_rate,
                            tag="Training/step_{}_{}_{}_reconstructed".format(step, label, i),
                        )
                        log(
                            train_logger,
                            audio=wav_prediction,
                            sampling_rate=sampling_rate,
                            tag="Training/step_{}_{}_{}_synthesized".format(step, label, i),
                        )

                if step % val_step == 0:
                    model.eval()
                    evaluate(model, step, configs, val_logger, vocoder,num2label=num2label)
                    model.train()

                if step % save_step == 0:
                    path = os.path.join(train_config["path"]["ckpt_path"], "{}.pth.tar".format(step))
                    print("save checkpoint at", path)
                    torch.save(
                        {
                            "model": model.state_dict(),
                            # "disc": disc.state_dict(),
                            "optimizer": optimizer._optimizer.state_dict(),
                            # "opt_d": opt_d._optimizer.state_dict(),
                        },
                        path
                    )

            if step == total_step:
                quit()
            step += 1
            outer_bar.update(1)

def read_gpu_ram():
    import nvidia_smi
    nvidia_smi.nvmlInit()
    handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)
    info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
    return info.total/1024/1024/1024

def rescale_batchsize(train_config):
    scale = int(read_gpu_ram() / 10) # 2080ti 12GB is the baseline
    train_config["optimizer"]["batch_size"] *= scale
    print("Use the batchsize of %s." % train_config["optimizer"]["batch_size"])
    return train_config
    
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

    # Read Config
    preprocess_config = yaml.load(
        open(args.preprocess_config, "r"), Loader=yaml.FullLoader
    )
    model_config = yaml.load(open(args.model_config, "r"), Loader=yaml.FullLoader)
    train_config = yaml.load(open(args.train_config, "r"), Loader=yaml.FullLoader)
      
    train_config = rescale_batchsize(train_config)  
    
    configs = (preprocess_config, model_config, train_config)

    n_gpus = torch.cuda.device_count()
    
    if n_gpus > 1:
        mp.spawn(main, nprocs=n_gpus, args=(n_gpus, args, configs),join=True)
    else:
        main(0, 1, args, configs)
        
    
