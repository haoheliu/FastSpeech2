import os
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import sys
sys.path.append("/vol/research/ai4sound/project/audio_generation/FastSpeech2")
import numpy as np
from transformer import Encoder, Decoder, PostNet
from model.modules import VarianceAdaptor, EnergyAdaptor
from utils.tools import get_mask_from_lengths
import model.unet as unet
import transformer.Constants as Constants
import model.glow.commons as commons
import model.glow.modules as modules
import model.glow.attentions as attentions

import model.wavenet.modules as modules_wavenet
import model.wavenet.commons as commons_wavenet

class WaveNetEncoder(nn.Module):
  def __init__(self,
      in_channels,
      out_channels,
      hidden_channels,
      kernel_size,
      dilation_rate,
      n_layers,
      gin_channels=0):
    super().__init__()
    self.in_channels = in_channels
    self.out_channels = out_channels
    self.hidden_channels = hidden_channels
    self.kernel_size = kernel_size
    self.dilation_rate = dilation_rate
    self.n_layers = n_layers
    self.gin_channels = gin_channels

    self.pre = nn.Conv1d(in_channels, hidden_channels, 1)
    self.enc = modules_wavenet.WN(hidden_channels, kernel_size, dilation_rate, n_layers, gin_channels=gin_channels)
    self.proj = nn.Conv1d(hidden_channels, out_channels, 1)

  def forward(self, x, x_mask, g=None):
    # x_mask = torch.unsqueeze(commons.sequence_mask(x_lengths, x.size(2)), 1).to(x.dtype)
    x = self.pre(x) * x_mask.unsqueeze(1)
    x = self.enc(x, x_mask.unsqueeze(1), g=g[..., None])
    stats = self.proj(x) * x_mask.unsqueeze(1)
    return stats.permute(0,2,1)
  
class FramePriorNet(torch.nn.Module):
  def __init__(self, in_channel, hidden_channel, out_channels, kernel_size=5, n_layers=4) -> None:
    super().__init__()
    self.conv = torch.nn.ModuleList()
    for idx in range(n_layers):
      if idx != 0:
        in_channel = hidden_channel
      self.conv += [torch.nn.Sequential(
        torch.nn.Conv1d(in_channel, hidden_channel, kernel_size, stride=1, padding=(kernel_size-1)//2),
        torch.nn.ReLU(),
        modules.LayerNorm(hidden_channel)
      )]
    self.proj = torch.nn.Conv1d(hidden_channel, out_channels * 2, 1)
    self.out_channels = out_channels
    self.smooth_length = 20
    self.pooling = torch.nn.AvgPool1d(kernel_size=self.smooth_length, stride=self.smooth_length // 2, padding=5)

  def smooth(self, x):
    # x: [bs, 64, 496]
    bs, mel_bin, mel_len = x.size()
    
    assert mel_len % self.smooth_length == 0, "%s %s" % (mel_len, self.smooth_length)
    
    x = self.pooling(x)
    x = x.unsqueeze(-1)
    x = x.expand(bs, mel_bin, (mel_len // self.smooth_length) * 2, self.smooth_length // 2)
    x = x.reshape(bs, mel_bin, mel_len)
    return x
    
  def forward(self, x, x_mask):
    # x_mask = torch.unsqueeze(commons.sequence_mask(x_lengths, x.size(2)), 1).to(x.dtype)
    for f in self.conv:
      x = f(x)
      x = x * x_mask
    x = self.smooth(x)
    # import ipdb; ipdb.set_trace()
    stats = self.proj(x) * x_mask # todo
    m, logs = torch.split(stats, self.out_channels, dim=1)
    return x, m, logs
  
class FlowSpecDecoder(nn.Module):
  def __init__(self, 
      in_channels, 
      hidden_channels, 
      kernel_size, 
      dilation_rate, 
      n_blocks, 
      n_layers, 
      p_dropout=0., 
      n_split=4,
      n_sqz=2,
      sigmoid_scale=False,
      gin_channels=0):
    super().__init__()

    self.in_channels = in_channels
    self.hidden_channels = hidden_channels
    self.kernel_size = kernel_size
    self.dilation_rate = dilation_rate
    self.n_blocks = n_blocks
    self.n_layers = n_layers
    self.p_dropout = p_dropout
    self.n_split = n_split
    self.n_sqz = n_sqz
    self.sigmoid_scale = sigmoid_scale
    self.gin_channels = gin_channels

    self.flows = nn.ModuleList()
    for b in range(n_blocks):
      self.flows.append(modules.ActNorm(channels=in_channels * n_sqz))
      self.flows.append(modules.InvConvNear(channels=in_channels * n_sqz, n_split=n_split))
      self.flows.append(
        attentions.CouplingBlock(
          in_channels * n_sqz,
          hidden_channels,
          kernel_size=kernel_size,  
          dilation_rate=dilation_rate,
          n_layers=n_layers,
          gin_channels=gin_channels,
          p_dropout=p_dropout,
          sigmoid_scale=sigmoid_scale
        )
      )

  def forward(self, x, x_mask, g=None, reverse=False):
    if not reverse:
      flows = self.flows
      logdet_tot = 0
    else:
      flows = reversed(self.flows)
      logdet_tot = None

    if self.n_sqz > 1:
      x, x_mask = commons.squeeze(x, x_mask, self.n_sqz)
    for f in flows:
      if not reverse:
        x, logdet = f(x, x_mask, g=g, reverse=reverse)
        logdet_tot += logdet
      else:
        x, logdet = f(x, x_mask, g=g, reverse=reverse)
    if self.n_sqz > 1:
      x, x_mask = commons.unsqueeze(x, x_mask, self.n_sqz)
    return x, logdet_tot

  def store_inverse(self):
    for f in self.flows:
      f.store_inverse()
      
class FastSpeech2(nn.Module):
    """ FastSpeech2 """

    def __init__(self, preprocess_config, model_config):
        super(FastSpeech2, self).__init__()
        self.model_config = model_config
        self.preprocess_config = preprocess_config
        
        self.target_length = self.preprocess_config["preprocessing"]["mel"]["target_length"]
        self.class_num = preprocess_config["class_num"]
        self.mel_bins = preprocess_config["preprocessing"]["mel"]["n_mel_channels"]
        
        self.speaker_emb = None
        # if model_config["multi_speaker"]:
        #     with open(
        #         os.path.join(
        #             preprocess_config["path"]["preprocessed_path"], "speakers.json"
        #         ),
        #         "r",
        #     ) as f:
        #         n_speaker = len(json.load(f)) + 1 # Add silence tokens
        #     self.speaker_emb = nn.Embedding(
        #         n_speaker,
        #         model_config["transformer"]["encoder_hidden"],
        #     )
        
        self.diff = DiffusionDecoder(unet_in_channels=2, N=model_config["N_diff_steps"])
        # self.diff_speaker_embedding = nn.Embedding(
        #         self.class_num,
        #         preprocess_config["preprocessing"]["mel"]["n_mel_channels"],
        #     )
        
        # self.energy_adaptor = EnergyAdaptor(preprocess_config, model_config)
        
        # self.proj_pitch = nn.Linear(model_config["transformer"]["encoder_hidden"], preprocess_config["preprocessing"]["mel"]["n_mel_channels"])
        # self.proj_energy = nn.Linear(model_config["transformer"]["encoder_hidden"], preprocess_config["preprocessing"]["mel"]["n_mel_channels"])
        
        
        self.label_proj = nn.Sequential(
          nn.Linear(self.class_num, 1024),
          nn.LeakyReLU(inplace=True),
          nn.Linear(1024, self.mel_bins)
        )
        
    def build_frame_energy_mask(self, logmels):
        energy = torch.sum(torch.exp(logmels), dim=-1) # [4, 496]
        cutoff = float(np.random.uniform(0.1,0.3))
        threshold = torch.max(energy, dim=1).values * cutoff
        return energy > threshold[:, None]
        
    def build_input_tokens(self, speakers, logmels):
        # speakers: [4,]
        # logmels: [4, 496, 64]
        # texts: [4, 62, 512]
        bs, t_len, mel_dim = logmels.size()
        tokens = speakers.unsqueeze(1).expand(speakers.size(0), t_len)
        valid_tokens = tokens * self.build_frame_energy_mask(logmels)
        # valid_tokens = nn.MaxPool1d(8, stride=8)(valid_tokens.float()).int()
        return valid_tokens
    
    def forward(
        self,
        mels,
        speakers, # (16,) The speaker id for each one in a batch, 
        gen=False
    ):  
        # low_bandwidth_mel = torch.randn_like(mels).to(mels.device)
        
        label_emb = self.label_proj(speakers)
        label_emb = label_emb.unsqueeze(1).expand(
            label_emb.size(0), self.target_length, label_emb.size(1)
        )
        
        if(not gen):
            diff_output = None
            diff_loss = self.diff(None, mels, g=label_emb, gen=False).mean() # TODO detach here
            postnet_output = mels
        else:
            diff_output = self.diff(None, mels, g=label_emb, gen=True) # TODO detach here
            postnet_output = diff_output
            diff_loss = None
        
        return diff_loss, postnet_output, mels
            
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
    # mean = torch.exp(log_mean_coeff[:, None, None]) * x + (1-torch.exp(log_mean_coeff[:, None, None]) ) * mu # remove mu
    mean = torch.exp(log_mean_coeff[:, None, None]) * x
    std = torch.sqrt(1. - torch.exp(2. * log_mean_coeff))
    return mean, std

  def cal_loss(self, x, mu, t, z, std, g=None):
    time_steps = t * (self.N - 1)
    if(mu is None):
      if g is not None:
          x = torch.stack([x, g], 1)
      else:
          x = torch.stack([x], 1)
    else:
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
        # y_T = torch.randn_like(mu) + mu # remove mu
        y_T = torch.randn_like(y) 
        y_t_plus_one = y_T
        y_t = None
        for n in tqdm(range(self.N - 1, 0, -1)):
          t = torch.FloatTensor(1).fill_(n).to(y.device)
          
          if(mu is None):
            if g is not None:
                x = torch.stack([y_t_plus_one, g], 1)
            else:
                x = torch.stack([y_t_plus_one], 1)
          else:
            if g is not None:
                x = torch.stack([y_t_plus_one, mu, g], 1)
            else:
                x = torch.stack([y_t_plus_one, mu], 1)
          grad = self.unet(x, t)
          # y_t = y_t_plus_one-0.5*self.delta_t*self.discrete_betas[n]*(mu-y_t_plus_one-grad)
          y_t = y_t_plus_one-0.5*self.delta_t*self.discrete_betas[n]*(-y_t_plus_one-grad)
          y_t_plus_one = y_t
      return y_t