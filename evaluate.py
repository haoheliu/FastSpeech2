import argparse
import os

import torch
import yaml
import torch.nn as nn
from torch.utils.data import DataLoader

from utils.model import get_model, get_vocoder
from utils.tools import to_device, log, synth_one_sample, synth_one_sample_val
from model import FastSpeech2Loss
from dataset import Dataset


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate(model, step, configs, logger=None, vocoder=None, pred_prosody=True, num2label={}):
    preprocess_config, model_config, train_config = configs

    fbank_mean = preprocess_config["preprocessing"]["mel"]["mean"]
    fbank_std = preprocess_config["preprocessing"]["mel"]["std"]
    
    def normalize(x):
        return (x-fbank_mean)/fbank_std

    def denormalize(x):
        return x * fbank_std + fbank_mean
    
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
    for batchs in loader:
        fbank, labels, fnames, waveform, seg_label = batchs 
        fbank = fbank.to(device)
        labels = labels.to(device)
        seg_label = seg_label.to(device)
        fbank = normalize(fbank)
        
        with torch.no_grad():
            # Forward
            diff_loss, generated = model(fbank, seg_label, gen=True)
            # generated = denormalize(generated)
        break
    
    if logger is not None:
        for i in range(fbank.size(0)):
            label = int(torch.where(labels[i] == 1)[0][0])
            label = num2label[label]
            func = synth_one_sample    
            fig, wav_reconstruction, wav_prediction = func(
                denormalize(fbank[i]),
                denormalize(generated[i]),
                labels[i],
                vocoder,
                model_config,
                preprocess_config,
            )
            log(logger, step)
            log(
                logger,
                fig=fig,
                tag="Validation/step_{}_{}_{}".format(step,label, i),
            )

            sampling_rate = preprocess_config["preprocessing"]["audio"]["sampling_rate"]
            
            log(
                    logger,
                    audio=waveform[i],
                    sampling_rate=sampling_rate,
                    tag="Training/step_{}_{}_{}_original".format(step, label, i),
            )
            log(
                logger,
                audio=wav_reconstruction,
                sampling_rate=sampling_rate,
                tag="Validation/step_{}_{}_reconstructed_{}".format(step, label, i),
            )
            log(
                logger,
                audio=wav_prediction,
                sampling_rate=sampling_rate,
                tag="Validation/step_{}_{}_synthesized_{}".format(step, label, i),
            )

    # return message


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--restore_step", type=int, default=30000)
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
    model = get_model(args, configs, device, train=False).to(device)

    message = evaluate(model, args.restore_step, configs)
    print(message)