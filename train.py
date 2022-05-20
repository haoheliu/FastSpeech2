import argparse
import os

import torch
import yaml
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from utils.model import get_model, get_vocoder, get_param_num, get_discriminator
from utils.tools import to_device, log, synth_one_sample
from model import FastSpeech2Loss
from dataset import Dataset
from evaluate import evaluate
import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def mle_lossfunc(z, m, logs, logdet, mask):
    # import ipdb; ipdb.set_trace()
    l = torch.sum(logs) + 0.5 * torch.sum(torch.exp(-2 * logs) * ((z - m)**2)) # neg normal likelihood w/o the constant term
    l = l - torch.sum(logdet) # log jacobian determinant
    l = l / torch.sum(torch.ones_like(z) * mask) # averaging across batch, channel and time axes
    l = l + 0.5 * math.log(2 * math.pi) # add the remaining constant term
    return l

def main(args, configs):
    print("Prepare training ...")

    preprocess_config, model_config, train_config = configs

    # Get dataset
    dataset = Dataset(
        "train.txt", preprocess_config, train_config, sort=True, drop_last=True
    )
    batch_size = train_config["optimizer"]["batch_size"]
    group_size = 4  # Set this larger than 1 to enable sorting in Dataset
    assert batch_size * group_size < len(dataset)
    loader = DataLoader(
        dataset,
        batch_size=batch_size * group_size,
        shuffle=True,
        collate_fn=dataset.collate_fn,
    )
    disc, opt_d = get_discriminator(args, configs, device, train=True)
    # Prepare model
    model, optimizer = get_model(args, configs, device, train=True)
    
    num_param = get_param_num(model)
    Loss = FastSpeech2Loss(preprocess_config, model_config).to(device)
    print("Number of FastSpeech2 Parameters:", num_param)

    # Load vocoder 
    vocoder = get_vocoder(model_config, device)

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
    epoch = 1
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
        inner_bar = tqdm(total=len(loader), desc="Epoch {}".format(epoch), position=1)
        # import ipdb; ipdb.set_trace()
        for batchs in loader:
            for batch in batchs:
                batch = to_device(batch, device)

                # Forward
                output, (z,m,logs,logdet,mel_masks) = model(*(batch[2:]), gen=False)

                # Cal Loss
                losses, (mel_predictions, mel_targets) = Loss(batch, output)
                mle_loss = mle_lossfunc(z,m,logs,logdet,~mel_masks.unsqueeze(1))
                
                disc_loss, fmap_loss = torch.tensor([0.0]), torch.tensor([0.0])
                r_losses, g_losses = torch.tensor([0.0]), torch.tensor([0.0])
                gen_loss = torch.tensor([0.0])
                
                if(step >2000000):
                    disc_real_outputs, _ = disc(mel_targets)
                    disc_generated_outputs, _ = disc(mel_predictions.detach())
                    disc_loss, r_losses, g_losses = Loss.discriminator_loss([disc_real_outputs], [disc_generated_outputs])
                    r_losses = torch.sum(torch.tensor(r_losses))
                    g_losses = torch.sum(torch.tensor(g_losses))
                    d_loss = 5 * disc_loss / grad_acc_step # 100 * 1
                    d_loss.backward()
                    if step % grad_acc_step == 0:
                        nn.utils.clip_grad_norm_(disc.parameters(), grad_clip_thresh)
                        opt_d.step_and_update_lr()
                        opt_d.zero_grad()
                
                if(step > 6000000):
                    # Backward
                    disc_real_outputs, fmap_real = disc(mel_targets)
                    disc_generated_outputs, fmap_generated = disc(mel_predictions)
                    fmap_loss = Loss.feature_loss([fmap_real], [fmap_generated])
                    gen_loss, gen_loss_items = Loss.generator_loss([disc_generated_outputs])
                    total_loss = losses[0] + fmap_loss + gen_loss
                else:
                    total_loss = losses[0]
                
                total_loss = total_loss + mle_loss
                total_loss = total_loss / grad_acc_step
                total_loss.backward()
                
                if step % grad_acc_step == 0:
                    # Clipping gradients to avoid gradient explosion
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip_thresh)
                    # Update weights
                    optimizer.step_and_update_lr()
                    optimizer.zero_grad()

                losses = list(losses)
                
                losses.extend([disc_loss, fmap_loss])
                losses.extend([r_losses, g_losses, gen_loss, mle_loss])
                
                if step % log_step == 0:
                    losses = [l.item() for l in losses]
                    message1 = "Step {}/{}, ".format(step, total_step)
                    message2 = "Total Loss: {:.4f}, Mel Loss: {:.4f}, Mel PostNet Loss: {:.4f}, Pitch Loss: {:.4f}, Energy Loss: {:.4f}, Duration Loss: {:.4f},  Disc Loss: {:.4f},  Fmap Loss: {:.4f}, r_loss: {:.4f}, g_loss: {:.4f}, Gen Loss: {:.4f}, MLE Loss: {:.4f}".format(
                        *losses
                    )

                    with open(os.path.join(train_log_path, "log.txt"), "a") as f:
                        f.write(message1 + message2 + "\n")

                    outer_bar.write(message1 + message2)
                    log(train_logger, step, losses=losses)

                if step % synth_step == 0:
                    with torch.no_grad():
                        model.eval()
                        output, (z,m,logs,logdet,mel_masks) = model(*(batch[2:]), gen=True)
                    model.train()
                    fig, wav_reconstruction, wav_prediction, tag = synth_one_sample(
                        batch,
                        output,
                        vocoder,
                        model_config,
                        preprocess_config,
                    )
                    log(
                        train_logger,
                        fig=fig,
                        tag="Training/step_{}_{}".format(step, tag),
                    )
                    sampling_rate = preprocess_config["preprocessing"]["audio"][
                        "sampling_rate"
                    ]
                    log(
                        train_logger,
                        audio=wav_reconstruction,
                        sampling_rate=sampling_rate,
                        tag="Training/step_{}_{}_reconstructed".format(step, tag),
                    )
                    log(
                        train_logger,
                        audio=wav_prediction,
                        sampling_rate=sampling_rate,
                        tag="Training/step_{}_{}_synthesized".format(step, tag),
                    )

                if step % val_step == 0:
                    model.eval()
                    message = evaluate(model, step, configs, val_logger, vocoder)
                    with open(os.path.join(val_log_path, "log.txt"), "a") as f:
                        f.write(message + "\n")
                    outer_bar.write(message)

                    model.train()

                if step % save_step == 0:
                    path = os.path.join(train_config["path"]["ckpt_path"], "{}.pth.tar".format(step))
                    print("save checkpoint at", path)
                    torch.save(
                        {
                            "model": model.state_dict(),
                            "disc": disc.state_dict(),
                            "optimizer": optimizer._optimizer.state_dict(),
                            "opt_d": opt_d._optimizer.state_dict(),
                        },
                        path
                    )

                if step == total_step:
                    quit()
                step += 1
                outer_bar.update(1)

            inner_bar.update(1)
        epoch += 1


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
    configs = (preprocess_config, model_config, train_config)

    main(args, configs)
