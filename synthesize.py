import re
import argparse
from string import punctuation
import librosa

import torch
import yaml
import numpy as np
from torch.utils.data import DataLoader

from _utils.model import get_model, get_vocoder
from dataset import Dataset
import pandas as pd
from _utils.tools import to_device, log, synth_one_sample
import os
import soundfile as sf
import time
from datetime import datetime

t = time.localtime()
date = datetime.today().strftime('%Y_%m_%d')
current_time = time.strftime("%H_%M_%S", t)
current_time = date +"_"+ current_time
# current_time="npy_synthesis_remove_0_05_TOP5_smooth_v2"

npy_PATH = "/mnt/fast/nobackup/users/hl01486/projects/general_audio_generation/text2label_t5"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def normalize(preprocess_config, x):
    fbank_mean = preprocess_config["preprocessing"]["mel"]["mean"]
    fbank_std = preprocess_config["preprocessing"]["mel"]["std"]
    return (x-fbank_mean)/fbank_std

def denormalize(preprocess_config, x):
    fbank_mean = preprocess_config["preprocessing"]["mel"]["mean"]
    fbank_std = preprocess_config["preprocessing"]["mel"]["std"]
    return x * fbank_std + fbank_mean
    
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

def synthesize_target_sound_npy(id, model, step, configs, vocoder, batchs, save_dir, num2label):
    preprocess_config, model_config, train_config = configs
    LABEL_T_SMOOTH=10
    fbank, labels, fnames, waveform, seg_label = batchs
    fbank = fbank.to(device)
    labels = labels.to(device)
    seg_label = seg_label.to(device)
    fbank = normalize(preprocess_config, fbank)

    for file in os.listdir(npy_PATH):
        
        if(".npy" not in file): continue
        
        # if(file != "A_wolf_is_singingnpy.npy"): continue
        
        print(file)
        label_name = file[:-4]
        fname = "[%s]gen_%s.wav" % (label_name, 0)
        
        if(os.path.exists(os.path.join(save_dir, "[%s]gen_%s.wav" % (label_name, 0)))): 
            print("exist", fname)
            continue
        
        with torch.no_grad():
            # Forward
            fpath = os.path.join(npy_PATH, file)
            
            seg_label = np.load(fpath)[0]
            # rank = np.mean(torch.sigmoid(torch.tensor(seg_label)).numpy(), axis=0)
            # rank = np.argsort(rank)
            
            seg_label = np.repeat(seg_label, 20, 0)
            seg_label = seg_label[:1056,:]
            seg_label = torch.sigmoid(torch.tensor(seg_label).cuda())
            # seg_label[:,rank[:-5]] *= 0
            ####################################
            # seg_label[:,153]=0.05
            ####################################
            seg_label = seg_label.expand(4,1056,527)
            
            seg_label[seg_label < 0.05] *= 1e-4    
            
            labels = torch.mean(seg_label,dim=1)
            
            diff_loss, generated, _ = model(fbank, labels, seg_label, gen=True)
            
            for i in range(fbank.size(0)):
                
                tag=""
                
                fig, wav_reconstruction, wav_prediction = synth_one_sample(
                    denormalize(preprocess_config,fbank[i]),
                    denormalize(preprocess_config,generated[i]),
                    tag,
                    vocoder,
                    model_config,
                    preprocess_config,
                )
                sf.write(file=os.path.join(save_dir, "[%s]gen_%s.wav" % (label_name, i)), data=wav_prediction, samplerate=preprocess_config["preprocessing"]["audio"]["sampling_rate"])


def synthesize_target_sound(id, model, step, configs, vocoder, batchs, save_dir, num2label):
    preprocess_config, model_config, train_config = configs

    fbank, labels, fnames, waveform, seg_label = batchs
    fbank = fbank.to(device)
    labels = labels.to(device)
    seg_label = seg_label.to(device)
    fbank = normalize(preprocess_config, fbank)
    
    # max_index = torch.argmax(torch.sum(seg_label, dim=1), dim=1)

    seg_label_bak = torch.zeros_like(seg_label).to(seg_label.device)
    labels_bak = torch.zeros_like(labels).to(labels.device)

    for label_id in range(0, 500):
        label_id = int(label_id)
        
        for k in range(1):
            seg_label = seg_label_bak.clone()
            labels = labels_bak.clone()
            
            with torch.no_grad():
                # Forward
                
                seg_label[:,:,label_id] = 1.0
                labels[:,label_id] = 1.0
                
                diff_loss, generated, _ = model(fbank, labels, seg_label, gen=True)
                
                for i in range(fbank.size(0)):
                    label_name = num2label[label_id]
                    
                    tag = ""
                    
                    fig, wav_reconstruction, wav_prediction = synth_one_sample(
                        denormalize(preprocess_config,fbank[i]),
                        denormalize(preprocess_config,generated[i]),
                        tag,
                        vocoder,
                        model_config,
                        preprocess_config,
                    )
                    sf.write(file=os.path.join(save_dir, "[%s]gen_%s_%s.wav" % (label_name, k, i)), data=wav_prediction, samplerate=preprocess_config["preprocessing"]["audio"]["sampling_rate"])

def synthesize_multiple_target_sound(id, model, step, configs, vocoder, batchs, save_dir, num2label):
    preprocess_config, model_config, train_config = configs
    
    # targets={0: 0.75, 1: 0.3, 137: 0.5}
    # targets={36: 0.5, 137: 0.8, 3: 0.4}
    # targets={137: 0.9, 138: 0.4, 152: 0.8, 153: 1.0, 283: 0.2}
    # targets={74: 0.9, 75: 0.5}
    targets={137: 1.0, 138: 1.0, 193:1.0}

    fbank, labels, fnames, waveform, seg_label = batchs
    fbank = fbank.to(device)
    labels = labels.to(device)
    seg_label = seg_label.to(device)
    fbank = normalize(preprocess_config, fbank)
    
    # max_index = torch.argmax(torch.sum(seg_label, dim=1), dim=1)

    seg_label_bak = torch.zeros_like(seg_label).to(seg_label.device)
    labels_bak = torch.zeros_like(labels).to(labels.device)
        
    for k in range(1):
        seg_label = seg_label_bak.clone()
        labels = labels_bak.clone()
        
        with torch.no_grad():
            # Forward
            
            for k in targets.keys():
                seg_label[:,:,k] = targets[k]
                labels[:,k] = targets[k]
            
            diff_loss, generated, _ = model(fbank, labels, seg_label, gen=True)
            
            for i in range(fbank.size(0)):
                label_name = [num2label[k] for k in targets.keys()]
                
                tag = ""
                
                fig, wav_reconstruction, wav_prediction = synth_one_sample(
                    denormalize(preprocess_config,fbank[i]),
                    denormalize(preprocess_config,generated[i]),
                    tag,
                    vocoder,
                    model_config,
                    preprocess_config,
                )
                sf.write(file=os.path.join(save_dir, "[%s]gen_%s_%s.wav" % (label_name, k, i)), data=wav_prediction, samplerate=preprocess_config["preprocessing"]["audio"]["sampling_rate"])


def synthesize_target_sound(id, model, step, configs, vocoder, batchs, save_dir, num2label):
    preprocess_config, model_config, train_config = configs

    fbank, labels, fnames, waveform, seg_label = batchs
    fbank = fbank.to(device)
    labels = labels.to(device)
    seg_label = seg_label.to(device)
    fbank = normalize(preprocess_config, fbank)
    
    # max_index = torch.argmax(torch.sum(seg_label, dim=1), dim=1)

    seg_label_bak = torch.zeros_like(seg_label).to(seg_label.device)
    labels_bak = torch.zeros_like(labels).to(labels.device)

    for label_id in range(0, 500):
        label_id = int(label_id)
        
        for k in range(1):
            seg_label = seg_label_bak.clone()
            labels = labels_bak.clone()
            
            with torch.no_grad():
                # Forward
                
                seg_label[:,:,label_id] = 1.0
                labels[:,label_id] = 1.0
                
                seg_label[:,:,0] = 1.0
                labels[:,0] = 1.0
                
                diff_loss, generated, _ = model(fbank, labels, seg_label, gen=True)
                
                for i in range(fbank.size(0)):
                    label_name = num2label[label_id]
                    
                    tag = ""
                    
                    fig, wav_reconstruction, wav_prediction = synth_one_sample(
                        denormalize(preprocess_config,fbank[i]),
                        denormalize(preprocess_config,generated[i]),
                        tag,
                        vocoder,
                        model_config,
                        preprocess_config,
                    )
                    sf.write(file=os.path.join(save_dir, "[%s]gen_%s_%s.wav" % (label_name, k, i)), data=wav_prediction, samplerate=preprocess_config["preprocessing"]["audio"]["sampling_rate"])



def synthesize_change_sound(id, model, step, configs, vocoder, batchs, save_dir, num2label):
    preprocess_config, model_config, train_config = configs

    fbank, labels, fnames, waveform, seg_label = batchs
    fbank = fbank.to(device)
    labels = labels.to(device)
    seg_label = seg_label.to(device)
    fbank = normalize(preprocess_config, fbank)
    
    max_index = torch.argmax(torch.sum(seg_label, dim=1), dim=1)

    seg_label_bak = seg_label.clone()
    labels_bak = labels.clone()

    jump_to = int(np.random.randint(low=5, high=25))

    for k in range(1):
        seg_label = seg_label_bak.clone()
        labels = labels_bak.clone()
        
        with torch.no_grad():
            # Forward
            ratio = 0.0         
            
            seg_label[:,:,max_index+jump_to] = seg_label[:,:,max_index]
            labels[:,max_index+jump_to] = labels[:, max_index]
            
            seg_label[:, max_index] *= ratio
            labels[:, max_index] *= ratio
            
            diff_loss, generated, _ = model(fbank, labels, seg_label, gen=True)
            
            for i in range(fbank.size(0)):
                label = [int(x) for x in torch.where(labels[i] == 1)[0]]
                label = [num2label[x] for x in label]
                max_ind = num2label[int(max_index[i]) + jump_to]
                orig_max_ind = num2label[int(max_index[i])]
                
                tag = ""
                for each in label:
                    tag += each+"_"
                
                fig, wav_reconstruction, wav_prediction = synth_one_sample(
                    denormalize(preprocess_config,fbank[i]),
                    denormalize(preprocess_config,generated[i]),
                    tag,
                    vocoder,
                    model_config,
                    preprocess_config,
                )
                sf.write(file=os.path.join(save_dir, "[%s]_[%s]_%sgen_%s.wav" % (orig_max_ind, max_ind, tag, k)), data=wav_prediction, samplerate=preprocess_config["preprocessing"]["audio"]["sampling_rate"])
                sf.write(file=os.path.join(save_dir, "[%s]_[%s]_%sreconstruct.wav" % (orig_max_ind, max_ind,tag)), data=wav_reconstruction, samplerate=preprocess_config["preprocessing"]["audio"]["sampling_rate"])


def synthesize_control(id, model, step, configs, vocoder, batchs, save_dir, num2label):
    preprocess_config, model_config, train_config = configs

    fbank, labels, fnames, waveform, seg_label = batchs
    fbank = fbank.to(device)
    labels = labels.to(device)
    seg_label = seg_label.to(device)
    fbank = normalize(preprocess_config, fbank)
    
    max_index = torch.argmax(torch.sum(seg_label, dim=1), dim=1)

    seg_label_bak = seg_label.clone()
    labels_bak = labels.clone()

    for k in range(5):
        seg_label = seg_label_bak.clone()
        labels = labels_bak.clone()
        
        with torch.no_grad():
            # Forward
            ratio = k*0.2            
            seg_label[:,:,max_index] *= ratio
            labels[:, max_index] *= ratio
            
            diff_loss, generated, _ = model(fbank, labels, seg_label, gen=True)
            
            for i in range(fbank.size(0)):
                label = [int(x) for x in torch.where(labels[i] == 1)[0]]
                label = [num2label[x] for x in label]
                max_ind = num2label[int(max_index[i])]
                tag = ""
                for each in label:
                    tag += each+"_"
                
                fig, wav_reconstruction, wav_prediction = synth_one_sample(
                    denormalize(preprocess_config,fbank[i]),
                    denormalize(preprocess_config,generated[i]),
                    tag,
                    vocoder,
                    model_config,
                    preprocess_config,
                )
                sf.write(file=os.path.join(save_dir, "%s_%sgen_%s.wav" % (max_ind, tag, k)), data=wav_prediction, samplerate=preprocess_config["preprocessing"]["audio"]["sampling_rate"])
                sf.write(file=os.path.join(save_dir, "%s_%sreconstruct.wav" % (max_ind,tag)), data=wav_reconstruction, samplerate=preprocess_config["preprocessing"]["audio"]["sampling_rate"])


def synthesize(id, model, step, configs, vocoder, batchs, save_dir, num2label):
    preprocess_config, model_config, train_config = configs

    fbank, labels, fnames, waveform, seg_label = batchs
    fbank = fbank.to(device)
    labels = labels.to(device)
    seg_label = seg_label.to(device)
    fbank = normalize(preprocess_config, fbank)
    
    for k in range(3):
        with torch.no_grad():
            # Forward
            
            diff_loss, generated, _ = model(fbank, labels, seg_label, gen=True)
            
            for i in range(fbank.size(0)):
                label = [int(x) for x in torch.where(labels[i] == 1)[0]]
                label = [num2label[x] for x in label]
                tag = ""
                for each in label:
                    tag += each+"_"
                
                fig, wav_reconstruction, wav_prediction = synth_one_sample(
                    denormalize(preprocess_config,fbank[i]),
                    denormalize(preprocess_config,generated[i]),
                    tag,
                    vocoder,
                    model_config,
                    preprocess_config,
                )
                sf.write(file=os.path.join(save_dir, "%sgen_%s.wav" % (tag, k)), data=wav_prediction, samplerate=preprocess_config["preprocessing"]["audio"]["sampling_rate"])
                sf.write(file=os.path.join(save_dir, "%sreconstruct.wav" % tag), data=wav_reconstruction, samplerate=preprocess_config["preprocessing"]["audio"]["sampling_rate"])

def label_transfer(id, model, step, configs, vocoder, batchs, save_dir, num2label):
    preprocess_config, model_config, train_config = configs

    fbank, labels, fnames, waveform, seg_label = batchs
    fbank = fbank.to(device)
    labels = labels.to(device)
    seg_label = seg_label.to(device)
    batchsize,tsize,fsize=fbank.size()
    fbank = normalize(preprocess_config, fbank)
    
    with torch.no_grad():
        # Forward
        mu = fbank[0].unsqueeze(0).expand(batchsize,tsize,fsize)
        # generated=mu
        diff_loss, generated, _ = model(fbank, labels, seg_label, gen=True, mu=mu)
        
        for i in range(fbank.size(0)):
            label = [int(x) for x in torch.where(labels[i] == 1)[0]]
            label = [num2label[x] for x in label]
            tag = ""
            for each in label:
                tag += each+"_"
            
            fig, wav_reconstruction, wav_prediction = synth_one_sample(
                denormalize(preprocess_config,fbank[i]),
                denormalize(preprocess_config,generated[i]),
                tag,
                vocoder,
                model_config,
                preprocess_config,
            )
            sf.write(file=os.path.join(save_dir, "%s_%sgen_%s.wav" % (id, tag, i)), data=wav_prediction, samplerate=preprocess_config["preprocessing"]["audio"]["sampling_rate"])
            sf.write(file=os.path.join(save_dir, "%s_%sreconstruct.wav" % (id, tag)), data=wav_reconstruction, samplerate=preprocess_config["preprocessing"]["audio"]["sampling_rate"])

            fig.savefig(os.path.join(save_dir, "%s_%s_%s.png" % (id, tag, i)))
    
    
if __name__ == "__main__":

    def seed_torch(seed=1029):
        import random
        print("Set seed to %s" % seed)
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) 
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    
    seed_torch(0)
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--restore_step", type=int, required=True)
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
    configs = (preprocess_config, model_config, train_config)

    # Get model
    model = get_model(args, "FastSpeech2",configs, device, train=False)

    # Load vocoder
    vocoder = get_vocoder(model_config, device, mel_bins=preprocess_config["preprocessing"]["mel"]["n_mel_channels"])
    
    _,_,num2label = build_id_to_label(preprocess_config["path"]["class_label_index"])
    fbank_mean = preprocess_config["preprocessing"]["mel"]["mean"]
    fbank_std = preprocess_config["preprocessing"]["mel"]["std"]
    
    save_dir = "synthesized/%s" % current_time
    print(save_dir)
    os.makedirs("synthesized/%s" % current_time, exist_ok=True)
    
    # Get dataset
    dataset = Dataset(
        preprocess_config, train_config, train=False
    )
    
    batch_size = train_config["optimizer"]["batch_size"]
    loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
    )

    # for id, batchs in enumerate(loader):
    #     # synthesize_target_sound(id, model, args.restore_step, configs, vocoder, batchs, save_dir, num2label)
    #     synthesize_multiple_target_sound(id, model, args.restore_step, configs, vocoder, batchs, save_dir, num2label)
    #     break

    # # Evaluation
    for id, batchs in enumerate(loader):
        # synthesize(id, model, args.restore_step, configs, vocoder, batchs, save_dir, num2label)
        # label_transfer(id, model, args.restore_step, configs, vocoder, batchs, save_dir, num2label)
        # synthesize_control(id, model, args.restore_step, configs, vocoder, batchs, save_dir, num2label)
        # synthesize_change_sound(id, model, args.restore_step, configs, vocoder, batchs, save_dir, num2label)
        synthesize_target_sound_npy(id, model, args.restore_step, configs, vocoder, batchs, save_dir, num2label)
        break