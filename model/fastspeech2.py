import os
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import sys
sys.path.append("/vol/research/ai4sound/project/audio_generation/FastSpeech2")

from transformer import Encoder, Decoder, PostNet
from model.modules import VarianceAdaptor
from utils.tools import get_mask_from_lengths
import model.unet as unet

class FastSpeech2(nn.Module):
    """ FastSpeech2 """

    def __init__(self, preprocess_config, model_config):
        super(FastSpeech2, self).__init__()
        self.model_config = model_config

        self.encoder = Encoder(model_config)
        self.variance_adaptor = VarianceAdaptor(preprocess_config, model_config)
        self.decoder = Decoder(model_config)
        self.mel_linear = nn.Linear(
            model_config["transformer"]["decoder_hidden"],
            preprocess_config["preprocessing"]["mel"]["n_mel_channels"],
        )
        self.postnet = PostNet(n_mel_channels=preprocess_config["preprocessing"]["mel"]["n_mel_channels"])

        self.speaker_emb = None
        if model_config["multi_speaker"]:
            with open(
                os.path.join(
                    preprocess_config["path"]["preprocessed_path"], "speakers.json"
                ),
                "r",
            ) as f:
                n_speaker = len(json.load(f))
            self.speaker_emb = nn.Embedding(
                n_speaker,
                model_config["transformer"]["encoder_hidden"],
            )
            self.diff_speaker_embedding = nn.Embedding(
                n_speaker,
                64,
            )
        self.diff = DiffusionDecoder(unet_in_channels=3)
        # self.proj= nn.Linear(model_config["transformer"]["encoder_hidden"], model_config["transformer"]["encoder_hidden"] * 2, 1)
        
    def forward(
        self,
        speakers, # (16,) The speaker id for each one in a batch, 
        texts, # (16,117), (batch_size, max_src_len) 
        src_lens, # (16,) The length of each one
        max_src_len, # 117, int
        mels=None, # (16, 845, 80)
        mel_lens=None, # (16,)
        max_mel_len=None, # 845, int
        p_targets=None,
        e_targets=None,
        d_targets=None,
        p_control=1.0,
        e_control=1.0,
        d_control=1.0,
        gen=False
    ):
        src_masks = get_mask_from_lengths(src_lens, max_src_len)
        mel_masks = (
            get_mask_from_lengths(mel_lens, max_mel_len)
            if mel_lens is not None
            else None
        )

        output = self.encoder(texts, src_masks)

        if self.speaker_emb is not None:
            g = self.speaker_emb(speakers)
            g_diff = self.diff_speaker_embedding(speakers)
            g_diff = g_diff.unsqueeze(1).expand(
                g_diff.size(0), max_mel_len, g_diff.size(1)
            )
            
            output = output + g.unsqueeze(1).expand(
                -1, max_src_len, -1
            )
        else: 
            g=None

        (
            output,
            p_predictions,
            e_predictions,
            log_d_predictions,
            d_rounded,
            mel_lens,
            mel_masks,
        ) = self.variance_adaptor(
            output,
            src_masks,
            mel_masks,
            max_mel_len,
            p_targets,
            e_targets,
            d_targets,
            p_control,
            e_control,
            d_control,
        )
        
        # stats = self.proj(output) * mel_masks.unsqueeze(-1)
        # m, logs = torch.split(stats, self.model_config["transformer"]["encoder_hidden"], dim=-1)
        # logs = torch.clamp(logs, min=0.05, max=None)
        # output = (m + torch.randn_like(m) * torch.exp(logs)) * mel_masks.unsqueeze(-1)
        output, mel_masks = self.decoder(output, mel_masks)
        output = self.mel_linear(output)
        postnet_output = self.postnet(output) + output # (16, 496, 64)
        
        if(not gen):
            diff_loss = self.diff(postnet_output, mels, g=g_diff, gen=False).mean() # TODO detach here
            diff_output = None
        else:
            diff_output = self.diff(postnet_output, mels, g=g_diff, gen=True) # TODO detach here
            diff_loss = None
        
        # diff_output = None
        # diff_loss = torch.tensor([0.0])
        
        return (
            output,
            postnet_output,
            p_predictions,
            e_predictions,
            log_d_predictions,
            d_rounded,
            src_masks,
            mel_masks,
            src_lens,
            mel_lens,
        ), (diff_output, diff_loss)
        
class DiffusionDecoder(nn.Module):
  def __init__(self, 
      n_speaker = 50,
      unet_channels=64,
      unet_in_channels=2,
      unet_out_channels=1,
      dim_mults=(1, 2, 4),
      groups=8,
      with_time_emb=True,
      beta_0=0.05,
      beta_1=20,
      N=1000,
      T=1):

    super().__init__()

    self.beta_0 = beta_0
    self.beta_1 = beta_1
    self.N = N
    self.T = T
    self.delta_t = T*1.0 / N
    self.discrete_betas = torch.linspace(beta_0, beta_1, N)
    self.unet = unet.Unet(dim=unet_channels, out_dim=unet_out_channels, dim_mults=dim_mults, groups=groups, channels=unet_in_channels, with_time_emb=with_time_emb)
    # self.linear = torch.nn.Linear(512, 64)

  def marginal_prob(self, mu, x, t):
    log_mean_coeff = -0.25 * t ** 2 * (self.beta_1 - self.beta_0) - 0.5 * t * self.beta_0
    mean = torch.exp(log_mean_coeff[:, None, None]) * x + (1-torch.exp(log_mean_coeff[:, None, None]) ) * mu
    std = torch.sqrt(1. - torch.exp(2. * log_mean_coeff))
    return mean, std

  def cal_loss(self, x, mu, t, z, std, g=None):
    time_steps = t * (self.N - 1)
    if g is not None:
        x = torch.stack([x, mu, g], 1)
    else:
        x = torch.stack([x, mu], 1)
    
    grad = self.unet(x, time_steps)
    loss = torch.square(grad + z / std[:, None, None]) * torch.square(std[:, None, None])
    return loss

  def forward(self, mu, y=None, g=None, gen=False):
    if not gen:
      t = torch.FloatTensor(y.shape[0]).uniform_(0, self.T-self.delta_t).to(y.device)+self.delta_t  # sample a random t
      mean, std = self.marginal_prob(mu, y, t)
      z = torch.randn_like(y)
      x = mean + std[:, None, None] * z
      loss = self.cal_loss(x, mu, t, z, std, g)
      return loss
    else:
      with torch.no_grad():
        y_T = torch.randn_like(mu) + mu
        y_t_plus_one = y_T
        y_t = None
        for n in tqdm(range(self.N - 1, 0, -1)):
          t = torch.FloatTensor(1).fill_(n).to(mu.device)
          if g is not None:
              x = torch.stack([y_t_plus_one, mu, g], 1)
          else:
              x = torch.stack([y_t_plus_one, mu], 1)
          grad = self.unet(x, t)
          y_t = y_t_plus_one-0.5*self.delta_t*self.discrete_betas[n]*(mu-y_t_plus_one-grad)
          y_t_plus_one = y_t
      return y_t

if __name__ == '__main__':
    model = DiffusionDecoder(unet_in_channels=3)
    mu=torch.randn(2,64, 496)
    y=torch.randn(2,64, 496)
    # mu=torch.randn(16,496,64)
    # y=torch.randn(16,496,64)
    g=torch.randn(2,64,496)
    print(model(mu, y, g=g).mean())