import os
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import sys
sys.path.append("/vol/research/ai4sound/project/audio_generation/FastSpeech2")

from transformer import Encoder, Decoder, PostNet
from model.modules import VarianceAdaptor, EnergyAdaptor
from utils.tools import get_mask_from_lengths
import model.unet as unet
import transformer.Constants as Constants
import model.glow.commons as commons
import model.glow.modules as modules
import model.glow.attentions as attentions

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
  
  def forward(self, x, x_mask):
    # x_mask = torch.unsqueeze(commons.sequence_mask(x_lengths, x.size(2)), 1).to(x.dtype)
    for f in self.conv:
      x = f(x)
      x = x * x_mask
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
          sigmoid_scale=sigmoid_scale))

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
        n_src_vocab = 50 + 1 # TODO hard code here on symbol numbers
        d_word_vec = model_config["transformer"]["encoder_hidden"]
        
        self.src_word_emb = nn.Embedding(
            n_src_vocab, d_word_vec, padding_idx=Constants.PAD
        )
                
        self.encoder = Encoder(model_config)
        self.latent_lstm = nn.LSTM(model_config["transformer"]["encoder_hidden"], model_config["transformer"]["encoder_hidden"], num_layers=1, batch_first=True)
        self.energy_adaptor = EnergyAdaptor(preprocess_config, model_config)
        
        hidden_dim=preprocess_config["preprocessing"]["mel"]["n_mel_channels"]
        self.mel_linear = nn.Linear(
            model_config["transformer"]["decoder_hidden"],
            preprocess_config["preprocessing"]["mel"]["n_mel_channels"],
        )
        self.frame_prior_net = FramePriorNet(hidden_dim, hidden_dim, hidden_dim)
        self.decoder = FlowSpecDecoder(hidden_dim, hidden_dim, kernel_size=5, dilation_rate=5, n_blocks=12, n_layers=4, p_dropout=0.0, n_split=4, n_sqz=1, sigmoid_scale=False, gin_channels=192)
      
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
                preprocess_config["preprocessing"]["mel"]["n_mel_channels"],
            )
        # self.proj= nn.Linear(model_config["transformer"]["encoder_hidden"], model_config["transformer"]["encoder_hidden"] * 2, 1)
        
    def build_frame_energy_mask(self, logmels):
        energy = torch.sum(torch.exp(logmels), dim=-1) # [4, 496]
        threshold = torch.max(energy, dim=1).values * 0.1
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
        mel_masks = (
            get_mask_from_lengths(mel_lens, max_mel_len)
            if mel_lens is not None
            else None
        )
        
        tokens = self.build_input_tokens(speakers, mels)
            
        tokens_emb = self.src_word_emb(tokens) 
        
        output,_ = self.latent_lstm(tokens_emb)
        
        (
            output,
            p_predictions,
            e_predictions,
            log_d_predictions,
            d_rounded,
            _,
            mel_masks,
        ) = self.energy_adaptor(
            output,
            mel_masks,
            mel_masks,
            max_mel_len,
            p_targets,
            e_targets,
            d_targets,
            p_control,
            e_control,
            d_control,
        )
        
        if self.speaker_emb is not None:
            g = self.speaker_emb(speakers)
            output = output + g.unsqueeze(1).expand(
                -1, max_mel_len, -1
            )
        else: 
            g=None
        
        output = self.encoder(output, mel_masks)
        output = self.mel_linear(output)
        output, m, logs = self.frame_prior_net(output.permute(0,2,1), ~mel_masks.unsqueeze(1))
        
        if(gen):
          z = (m + torch.exp(logs) * torch.randn_like(m)) * (~mel_masks.unsqueeze(1))
          mel_pred, logdet = self.decoder(z, ~mel_masks.unsqueeze(1), g=g.unsqueeze(-1), reverse=True)
          postnet_output = mel_pred.permute(0,2,1)
          
        else:
          z, logdet = self.decoder(mels.permute(0,2,1), ~mel_masks.unsqueeze(1), g=g.unsqueeze(-1), reverse=False)
          postnet_output = mels
          # z_rand = (m + torch.exp(logs) * torch.randn_like(m)) * (~mel_masks.unsqueeze(1))
          # mel_pred, _ = self.decoder(z_rand, ~mel_masks.unsqueeze(1), g=g.unsqueeze(-1), reverse=True)
          
        # mel_pred = mel_pred.permute(0,2,1)
        # postnet_output = self.postnet(mel_pred) + mel_pred
        
        return (
            output,
            postnet_output,
            p_predictions,
            e_predictions,
            log_d_predictions,
            d_rounded,
            mel_masks,
            mel_masks,
            src_lens,
            mel_lens,
        ), (z,m,logs,logdet,mel_masks)