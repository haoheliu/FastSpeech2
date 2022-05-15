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
import transformer.Constants as Constants
import model.wavenet.modules as modules
import model.wavenet.commons as commons

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
    self.enc = modules.WN(hidden_channels, kernel_size, dilation_rate, n_layers, gin_channels=gin_channels)
    self.proj = nn.Conv1d(hidden_channels, out_channels, 1)

  def forward(self, x, x_mask, g=None):
    # x_mask = torch.unsqueeze(commons.sequence_mask(x_lengths, x.size(2)), 1).to(x.dtype)
    x = self.pre(x) * x_mask.unsqueeze(1)
    x = self.enc(x, x_mask.unsqueeze(1), g=g[..., None])
    stats = self.proj(x) * x_mask.unsqueeze(1)
    return stats.permute(0,2,1)

class WaveNet(nn.Module):
    """ WaveNet """

    def __init__(self, preprocess_config, model_config):
        super(WaveNet, self).__init__()
        self.model_config = model_config
        n_src_vocab = 50 + 1 # TODO hard code here on symbol numbers
        d_word_vec = model_config["transformer"]["encoder_hidden"]

        self.src_word_emb = nn.Embedding(
            n_src_vocab, d_word_vec, padding_idx=Constants.PAD
        )
        
        self.wn = WaveNetEncoder(in_channels=512, out_channels=512, hidden_channels=512, kernel_size=5, dilation_rate=1, n_layers=16, gin_channels=512)
                
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
                preprocess_config["preprocessing"]["mel"]["n_mel_channels"],
            )
        
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
        return valid_tokens # Frame level tokens
    
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
        bs=speakers.size(0)
        
        #################################### label only ##################################################
        texts = None
        mel_masks = (
            get_mask_from_lengths(mel_lens, max_mel_len)
            if mel_lens is not None
            else None
        )
        src_masks = mel_masks
        
        if(mels is not None):
            tokens = self.build_input_tokens(speakers, mels)
        else:
            tokens = texts

        if self.speaker_emb is not None:
            g = self.speaker_emb(speakers)
        else: 
            g=None
            
        tokens_emb = self.src_word_emb(tokens) # TODO We havn't applied mask yet    
        output = self.wn(tokens_emb.permute(0,2,1), ~mel_masks, g=g) 
    
        if(g is not None):
            output = output + g.unsqueeze(1).expand(
                -1, max_mel_len, -1
            )
    
        if self.speaker_emb is not None:
            g = self.speaker_emb(speakers)
            output = output + g.unsqueeze(1).expand(
                -1, max_mel_len, -1
            )
        else: 
            g=None
        output = self.mel_linear(output)
        postnet_output = self.postnet(output) + output # (16, 496, 64)
        ######################################## Dirty things ##############################################
        diff_loss, latent_loss = torch.tensor([0.0]).cuda(), torch.tensor([0.0]).cuda()
        p_predictions,e_predictions,log_d_predictions,d_rounded = torch.zeros((bs, 62)).cuda(), torch.zeros((bs, 62)).cuda(), torch.zeros((bs, 62)).cuda(), torch.zeros((bs, 62)).cuda()
        diff_output = None
        ####################################################################################################
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
        ), (diff_output, diff_loss, latent_loss)
        
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
        self.latent_encoder = Encoder(model_config)
        self.latent_lstm = nn.LSTM(512, 512, num_layers=1, batch_first=True)
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
                preprocess_config["preprocessing"]["mel"]["n_mel_channels"],
            )
        self.diff = DiffusionDecoder(unet_in_channels=3)
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
        valid_tokens = nn.MaxPool1d(8, stride=8)(valid_tokens.float()).int()
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
        
        #################################### label only ##################################################
        texts = None
        src_masks = get_mask_from_lengths(src_lens, max_src_len)
        mel_masks = (
            get_mask_from_lengths(mel_lens, max_mel_len)
            if mel_lens is not None
            else None
        )
        if(mels is not None):
            tokens = self.build_input_tokens(speakers, mels)
        else:
            tokens = texts

        tokens_emb = self.src_word_emb(tokens) # TODO We havn't applied mask yet
        latent_prediction,_ = self.latent_lstm(tokens_emb)
        latent_prediction = self.latent_encoder(latent_prediction, src_masks)       
        
        if(gen):
            output = self.encoder(latent_prediction, src_masks)
            max_mel_len = 8 * max_src_len
        else:
            output = self.encoder(latent_prediction, src_masks)
        latent_loss = torch.tensor([0.0]).cuda()
        
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
        ###################################################################################################
        #################################### with embeddings ##################################################
        # src_masks = get_mask_from_lengths(src_lens, max_src_len)
        
        # mel_masks = (
        #     get_mask_from_lengths(mel_lens, max_mel_len)
        #     if mel_lens is not None
        #     else None
        # )
        # if(mels is not None):
        #     tokens = self.build_input_tokens(speakers, mels)
        # else:
        #     tokens = texts
            
        # tokens_emb = self.src_word_emb(tokens) # TODO We havn't applied mask yet
        # latent_prediction,_ = self.latent_lstm(tokens_emb)
        # latent_prediction = self.latent_encoder(latent_prediction, src_masks)       
        
        # if(gen):
        #     output = self.encoder(latent_prediction, src_masks)
        #     latent_loss = torch.tensor([0.0])
        #     max_mel_len = 8 * max_src_len
        # else:
        #     output = self.encoder(texts, src_masks)
        #     latent_loss = torch.abs(texts - latent_prediction).mean()
        
        # if self.speaker_emb is not None:
        #     g = self.speaker_emb(speakers)
        #     g_diff = self.diff_speaker_embedding(speakers)
        #     g_diff = g_diff.unsqueeze(1).expand(
        #         g_diff.size(0), max_mel_len, g_diff.size(1)
        #     )
            
        #     output = output + g.unsqueeze(1).expand(
        #         -1, max_src_len, -1
        #     )
        # else: 
        #     g=None
        ###################################################################################################
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
        
        output, mel_masks = self.decoder(output, mel_masks)
        output = self.mel_linear(output)
        postnet_output = self.postnet(output) + output # (16, 496, 64)
        
        if(not gen):
            diff_loss = self.diff(postnet_output, mels, g=g_diff, gen=False).mean() # TODO detach here
            diff_output = None
        else:
            diff_output = self.diff(postnet_output, mels, g=g_diff, gen=True) # TODO detach here
            diff_loss = None
        return (
            output,
            postnet_output,
            p_predictions, # [bs, 62]
            e_predictions, # [bs, 62]
            log_d_predictions, # [bs, 62]
            d_rounded, # [bs, 62]
            src_masks,
            mel_masks,
            src_lens,
            mel_lens,
        ), (diff_output, diff_loss, latent_loss)
        
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