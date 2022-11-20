import os
import json

import torch
import numpy as np

import hifigan
from model import FastSpeech2, ScheduledOptim, Label2Audio, Label2Audiov2
from discriminator import SpecDiscriminator

def get_model(args, model_name, configs, device, train=False):
    (preprocess_config, model_config, train_config) = configs

    model = eval(model_name)(preprocess_config, model_config).to(device)
    if args.restore_step:
        ckpt_path = os.path.join(
            train_config["path"]["ckpt_path"],
            "{}.pth.tar".format(args.restore_step),
        )
        ckpt = torch.load(ckpt_path)
        state_dict = ckpt["model"]
        new_state_dict = {}
        for k in state_dict.keys():
            if("module." in k):
                new_k = k[7:]
                new_state_dict[new_k] = state_dict[k]
            else:
                new_state_dict[k] = state_dict[k]
                
        model.load_state_dict(new_state_dict)

    if train:
        scheduled_optim = ScheduledOptim(
            model, train_config, model_config, args.restore_step
        )
        if args.restore_step:
            scheduled_optim.load_state_dict(ckpt["optimizer"])
        model.train()
        return model, scheduled_optim

    model.eval()
    model.requires_grad_ = False
    return model

def get_discriminator(args, configs, device, train=False):
    (preprocess_config, model_config, train_config) = configs

    model = SpecDiscriminator().to(device)
    # if args.restore_step:
    #     ckpt_path = os.path.join(
    #         train_config["path"]["ckpt_path"],
    #         "{}.pth.tar".format(args.restore_step),
    #     )
    #     ckpt = torch.load(ckpt_path)
    #     model.load_state_dict(ckpt["model"])

    if train:
        scheduled_optim = ScheduledOptim(
            model, train_config, model_config, args.restore_step
        )
        # if args.restore_step:
        #     scheduled_optim.load_state_dict(ckpt["optimizer"])
        model.train()
        return model, scheduled_optim

    model.eval()
    model.requires_grad_ = False
    return model


def get_param_num(model):
    num_param = sum(param.numel() for param in model.parameters())
    return num_param


def get_vocoder(config, device, mel_bins):
    name = config["vocoder"]["model"]
    speaker = config["vocoder"]["speaker"]

    if name == "MelGAN":
        if speaker == "LJSpeech":
            vocoder = torch.hub.load(
                "descriptinc/melgan-neurips", "load_melgan", "linda_johnson"
            )
        elif speaker == "universal":
            vocoder = torch.hub.load(
                "descriptinc/melgan-neurips", "load_melgan", "multi_speaker"
            )
        vocoder.mel2wav.eval()
        vocoder.mel2wav.to(device)
    elif name == "HiFi-GAN":
        if(mel_bins==64):
            with open("hifigan/config_16k_64.json", "r") as f:
                config = json.load(f)
            config = hifigan.AttrDict(config)
            vocoder = hifigan.Generator(config)
            print("Load hifigan/g_01080000")
            ckpt = torch.load("/mnt/fast/nobackup/users/hl01486/projects/general_audio_generation/conditional_transfer/FastSpeech2/hifigan/g_01080000")
            vocoder.load_state_dict(ckpt["generator"])
            vocoder.eval()
            vocoder.remove_weight_norm()
            vocoder.to(device)
        elif(mel_bins==128):
            with open("hifigan/config_16k_128.json", "r") as f:
                config = json.load(f)
            config = hifigan.AttrDict(config)
            vocoder = hifigan.Generator(config)
            print("Load hifigan/g_01440000")
            ckpt = torch.load("/mnt/fast/nobackup/users/hl01486/projects/general_audio_generation/conditional_transfer/FastSpeech2/hifigan/g_01440000")
            vocoder.load_state_dict(ckpt["generator"])
            vocoder.eval()
            vocoder.remove_weight_norm()
            vocoder.to(device)
    return vocoder


def vocoder_infer(mels, vocoder, model_config, preprocess_config, lengths=None):
    name = model_config["vocoder"]["model"]
    with torch.no_grad():
        if name == "MelGAN":
            wavs = vocoder.inverse(mels / np.log(10))
        elif name == "HiFi-GAN":
            wavs = vocoder(mels).squeeze(1)

    wavs = (
        wavs.cpu().numpy()
        * preprocess_config["preprocessing"]["audio"]["max_wav_value"]
    ).astype("int16")
    wavs = [wav for wav in wavs]

    for i in range(len(mels)):
        if lengths is not None:
            wavs[i] = wavs[i][: lengths[i]]

    return wavs
