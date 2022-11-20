import re
import argparse
from string import punctuation
import librosa

import torch
import yaml
import numpy as np
from torch.utils.data import DataLoader

from utils.model import get_model, get_vocoder
from dataset import Dataset
import pandas as pd
from utils.tools import to_device, log, synth_one_sample
import os
import soundfile as sf
import time
from datetime import datetime

t = time.localtime()
date = datetime.today().strftime('%Y_%m_%d')
current_time = time.strftime("%H_%M_%S", t)
current_time = date +"_"+ current_time
# current_time="npy_synthesis_remove_0_05_TOP5_smooth_v2"

# npy_PATH = "/mnt/fast/nobackup/users/hl01486/projects/general_audio_generation/text2label_t5"

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
    
def style_transfer(dataset, filename, model, step, configs, vocoder, batchs, save_dir, num2label):
    id="style_transfer"
    fbank, aug_fbank = dataset.extract_feature(filename)
    fbank = fbank.to(device)[None,...]
    aug_fbank = aug_fbank.to(device)[None,...]
    labels = None
    
    seg_label = np.zeros((53, 527))
    
    seg_label[:,0] = 1.0
    seg_label[:,1] = 1.0
    
    seg_label[:,137] = 1.0
    
    seg_label = np.repeat(seg_label, 20, 0)
    seg_label = seg_label[:1056,:]
    seg_label = torch.tensor(seg_label).to(device).float()
    seg_label = seg_label[None,...]
                
    with torch.no_grad():
        # Forward
        diff_loss, generated, _ = model(fbank, aug_fbank, labels, seg_label, gen=True)
        
        for i in range(fbank.size(0)):
            tag = "tag"
            fig, wav_reconstruction, wav_prediction = synth_one_sample(
                # denormalize(preprocess_config,fbank[i]),
                denormalize(preprocess_config,aug_fbank[i]),
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
    model = get_model(args, "Label2Audiov2", configs, device, train=False)

    # Load vocoder
    vocoder = get_vocoder(model_config, device, mel_bins=preprocess_config["preprocessing"]["mel"]["n_mel_channels"])
    
    _,_,num2label = build_id_to_label(preprocess_config["path"]["class_label_index"])
    fbank_mean = preprocess_config["preprocessing"]["mel"]["mean"]
    fbank_std = preprocess_config["preprocessing"]["mel"]["std"]
    
    save_dir = "synthesized/%s" % current_time
    os.makedirs("synthesized/%s" % current_time, exist_ok=True)
    
    # Get dataset
    dataset = Dataset(
        preprocess_config, train_config, train=False
    )
    
    batch_size = train_config["optimizer"]["batch_size"]
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
    )

    # Evaluation
    # for id, batchs in enumerate(loader):
        # synthesize(id, model, args.restore_step, configs, vocoder, batchs, save_dir, num2label)
        # label_transfer(id, model, args.restore_step, configs, vocoder, batchs, save_dir, num2label)
        # synthesize_control(id, model, args.restore_step, configs, vocoder, batchs, save_dir, num2label)
        # synthesize_change_sound(id, model, args.restore_step, configs, vocoder, batchs, save_dir, num2label)
        # label_transfer(id, model, args.restore_step, configs, vocoder, batchs, save_dir, num2label)
    
    fname = "data/audio/speech_test_1.wav"   
    style_transfer(dataset, fname, model, args.restore_step, configs, vocoder, None, save_dir, num2label)