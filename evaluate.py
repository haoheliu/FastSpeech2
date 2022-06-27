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
last_pitch = None
last_energy = None

def evaluate(model, step, configs, logger=None, vocoder=None, pred_prosody=True):
    global last_pitch, last_energy
    preprocess_config, model_config, train_config = configs

    # Get dataset
    dataset = Dataset(
        "val.txt", preprocess_config, train_config, sort=False, drop_last=False
    )
    batch_size = train_config["optimizer"]["batch_size"]
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        collate_fn=dataset.collate_fn,
    )

    # Get loss function
    Loss = FastSpeech2Loss(preprocess_config, model_config).to(device)

    # Evaluation
    loss_sums = [0 for _ in range(6)]
    idx = 0

    def range_compress(data, rate=0.5):
        min_val = torch.min(data)
        data = data - min_val
        data = data * rate
        data += min_val
        return data

    for batchs in loader:
        if(idx > 2): break # Run only one sample
        for batch in batchs: 
            idx += 1
            # Remove the target pitch and energy
            batch = to_device(batch, device)

            # batch = list(batch)
            # batch[9] = torch.zeros_like(batch[9]).to(batch[9].device)
            # batch[9] = range_compress(batch[9], 0.0)
            # batch[10] = range_compress(batch[10], 1.0)  

            # if(last_energy is None and last_pitch is None):
            #     last_pitch = batch[9]
            #     last_energy = batch[10]

            # if(last_pitch.size() == batch[9].size()):
            #     batch[9] = (last_pitch + batch[9]) / 2
            #     batch[10] = (last_energy + batch[10]) / 2


            # from scipy.signal import medfilt
            # import numpy as np
            # for i in range(batch[9].size(0)):
            #     batch[9][i] = torch.tensor(medfilt(batch[9][i].detach().cpu().numpy(), kernel_size=7)).cuda()
            #     batch[10][i] = torch.tensor(medfilt(batch[10][i].detach().cpu().numpy(), kernel_size=7)).cuda()

            # batch = tuple(batch)

            if(pred_prosody):
                batch = list(batch)
                batch[9] = None
                batch[10] = None
                batch = tuple(batch)

            with torch.no_grad():
                # Forward
                output, _ = model(*(batch[2:]), gen=True)
                # Cal Loss
                # losses,_ = Loss(batch, output)

                # for i in range(len(losses)):
                #     loss_sums[i] += losses[i].item() * len(batch[0])
    # loss_means = [loss_sum / len(dataset) for loss_sum in loss_sums]

    # message = "Validation Step {}, Total Loss: {:.4f}, Mel Loss: {:.4f}, Mel PostNet Loss: {:.4f}, Pitch Loss: {:.4f}, Energy Loss: {:.4f}, Duration Loss: {:.4f}".format(
    #     *([step] + [l for l in loss_means])
    # )

    if logger is not None:

        func = synth_one_sample_val if(pred_prosody) else synth_one_sample    
        fig, wav_reconstruction, wav_prediction, tag = func(
            batch,
            output,
            vocoder,
            model_config,
            preprocess_config,
        )
        log(logger, step)
        log(
            logger,
            fig=fig,
            tag="Validation/step_{}_{}_{}".format(step, tag, pred_prosody),
        )
        sampling_rate = preprocess_config["preprocessing"]["audio"]["sampling_rate"]
        log(
            logger,
            audio=wav_reconstruction,
            sampling_rate=sampling_rate,
            tag="Validation/step_{}_{}_reconstructed_{}".format(step, tag, pred_prosody),
        )
        log(
            logger,
            audio=wav_prediction,
            sampling_rate=sampling_rate,
            tag="Validation/step_{}_{}_synthesized_{}".format(step, tag, pred_prosody),
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