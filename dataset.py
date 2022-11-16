# Author: David Harwath
# with some functions borrowed from https://github.com/SeanNaren/deepspeech.pytorch
import csv
import json
import wave
import torchaudio
import numpy as np
import torch
import torch.nn.functional
from torch.utils.data import Dataset
import random
import audio as Audio
import librosa
import os
import torchvision

def make_index_dict(label_csv):
    index_lookup = {}
    with open(label_csv, 'r') as f:
        csv_reader = csv.DictReader(f)
        line_count = 0
        for row in csv_reader:
            index_lookup[row['mid']] = row['index']
            line_count += 1
    return index_lookup

def make_name_dict(label_csv):
    name_lookup = {}
    with open(label_csv, 'r') as f:
        csv_reader = csv.DictReader(f)
        line_count = 0
        for row in csv_reader:
            name_lookup[row['index']] = row['display_name']
            line_count += 1
    return name_lookup

def lookup_list(index_list, label_csv):
    label_list = []
    table = make_name_dict(label_csv)
    for item in index_list:
        label_list.append(table[item])
    return label_list

def preemphasis(signal,coeff=0.97):
    """perform preemphasis on the input signal.
    :param signal: The signal to filter.
    :param coeff: The preemphasis coefficient. 0 is none, default 0.97.
    :returns: the filtered signal.
    """
    return np.append(signal[0],signal[1:]-coeff*signal[:-1])

class Dataset(Dataset):
    def __init__(self, preprocess_config, train_config, train=True, with_context_prob=0.5):
        """
        Dataset that manages audio recordings
        :param audio_conf: Dictionary containing the audio loading and preprocessing settings
        :param dataset_json_file
        """
        self.preprocess_config = preprocess_config
        self.train_config = train_config
        self.datapath = preprocess_config["path"]["train_data"] if(train) else preprocess_config["path"]["test_data"]
        with open(self.datapath, 'r') as fp:
            data_json = json.load(fp)

        self.data = data_json['data']
        np.random.shuffle(self.data)
        self.melbins = preprocess_config["preprocessing"]["mel"]["n_mel_channels"]
        self.freqm = preprocess_config["preprocessing"]["mel"]["freqm"]
        self.timem = preprocess_config["preprocessing"]["mel"]["timem"]
        self.mixup = train_config["augmentation"]["mixup"]
        self.dataset = preprocess_config['dataset']
        self.sampling_rate = preprocess_config["preprocessing"]["audio"]["sampling_rate"]
        self.segment_label_path = preprocess_config["path"]["segment_label_path"]
        self.target_length = self.preprocess_config["preprocessing"]["mel"]["target_length"]
        self.use_blur = self.preprocess_config["preprocessing"]["mel"]["blur"]
        self.with_context_prob = with_context_prob
        self.feature_cache_path = self.preprocess_config["path"]["feature_save_path"]
        
        print("Use mixup rate of %s; Use SpecAug (T,F) of (%s, %s); Use blurring effect or not %s" % (self.mixup, self.timem, self.freqm, self.use_blur))
        
        print('now process ' + self.dataset)
        # dataset spectrogram mean and std, used to normalize the input
        self.norm_mean = preprocess_config["preprocessing"]["mel"]["mean"]
        self.norm_std = preprocess_config["preprocessing"]["mel"]["std"]
        
        # skip_norm is a flag that if you want to skip normalization to compute the normalization stats using src/get_norm_stats.py, if Ture, input normalization will be skipped for correctly calculating the stats.
        # set it as True ONLY when you are getting the normalization stats.
        self.skip_norm = False
        self.noise = False
        if self.noise == True:
            print('now use noise augmentation')

        self.index_dict = make_index_dict(preprocess_config["path"]["class_label_index"])
        self.label_num = len(self.index_dict)
        print('number of classes is {:d}'.format(self.label_num))

        self.STFT = Audio.stft.TacotronSTFT(
            preprocess_config["preprocessing"]["stft"]["filter_length"],
            preprocess_config["preprocessing"]["stft"]["hop_length"],
            preprocess_config["preprocessing"]["stft"]["win_length"],
            preprocess_config["preprocessing"]["mel"]["n_mel_channels"],
            preprocess_config["preprocessing"]["audio"]["sampling_rate"],
            preprocess_config["preprocessing"]["mel"]["mel_fmin"],
            preprocess_config["preprocessing"]["mel"]["mel_fmax"],
        )
        
        self.fbank_mean = preprocess_config["preprocessing"]["mel"]["mean"]
        self.fbank_std = preprocess_config["preprocessing"]["mel"]["std"]
    
    def normalize(self, x):
        return (x-self.fbank_mean)/self.fbank_std
    
    def resample(self, waveform, sr):
        if(sr==32000 and self.sampling_rate==16000):
            waveform = waveform[::2]
        return waveform

    def _wav2fbank(self, filename):
        # basename = os.path.basename(filename)[:-4]
        # cache_path = os.path.join(self.feature_cache_path, "%s.npy" % basename)
        # if(not os.path.exists(cache_path)):
        
        waveform, sr = librosa.load(filename, sr=None, mono=True)
        waveform = self.resample(waveform, sr)
        waveform = waveform - np.mean(waveform)
        waveform = waveform[None,...]
        waveform_length = int(self.target_length * self.preprocess_config["preprocessing"]["stft"]["hop_length"])
        
        if waveform_length > waveform.shape[1]:
            # padding
            temp_wav = np.zeros((1, waveform_length))
            temp_wav[:, :waveform.shape[1]] = waveform
            waveform = temp_wav
        else:
            # cutting
            waveform = waveform[:, :waveform_length]
            
        waveform = waveform[0,...]          
        
        fbank, _ = Audio.tools.get_mel_from_wav(waveform, self.STFT)
            # np.save(cache_path, fbank)
        # else:
        #     fbank = np.load(cache_path)

        fbank = torch.FloatTensor(fbank.T)
        
        n_frames = fbank.shape[0]

        p = self.target_length - n_frames
        # cut and pad
        if p > 0:
            m = torch.nn.ZeroPad2d((0, 0, 0, p))
            fbank = m(fbank)
        elif p < 0:
            fbank = fbank[0:self.target_length, :]
            
        return fbank, 0
        
    def decide_with_context_or_not(self):
        coin = self.random_uniform(0,1)
        return coin < self.with_context_prob
        
    def __getitem__(self, index):
        """
        returns: image, audio, nframes
        where image is a FloatTensor of size (3, H, W)
        audio is a FloatTensor of size (N_freq, N_frames) for spectrogram, or (N_frames) for waveform
        nframes is an integer
        """

        datum = self.data[index]
        label_indices = np.zeros(self.label_num)
        fbank, mix_lambda = self._wav2fbank(datum['wav'])
        for label_str in datum['labels'].split(','):
            label_indices[int(self.index_dict[label_str])] = 1.0

        label_indices = torch.FloatTensor(label_indices)

        # SpecAug, not do for eval set
        assert torch.min(fbank) < 0
        fbank = fbank.exp()
        
        original_fbank = (fbank+1e-7).log().clone()
        ############################### Blur and Spec Aug ####################################################
        fbank = torch.transpose(fbank, 0, 1)
        fbank = fbank.unsqueeze(0)
        
        # fbank_with_context = self.decide_with_context_or_not()
        
        # if(not fbank_with_context):
        #     fbank = fbank * 0.0 # Without any context information
        # else:
        if(self.use_blur): 
            fbank = self.blur(fbank)
        if self.freqm != 0:
            fbank = self.frequency_masking(fbank, self.freqm)
        if self.timem != 0:
            fbank = self.time_masking(fbank, self.timem)
        
        fbank = (fbank+1e-7).log()

        fbank = fbank.squeeze(0)
        fbank = torch.transpose(fbank, 0, 1)
        
        ##############segment label##############
        while(True):
            try:
                seg_label_fname = os.path.basename(datum['wav']).replace(".wav",".npy")
                seg_label_fpath = os.path.join(self.segment_label_path, seg_label_fname)
                seg_label = np.load(seg_label_fpath)
                seg_label = np.repeat(seg_label, 20, 0)
                seg_label = seg_label[:self.target_length,:]
                break
            except Exception as e:
                print(e)
                if(index == len(self.data)-1):
                    index = 0
                datum = self.data[index+1]
                
        #########################################
        original_fbank = self.normalize(original_fbank)
        fbank = self.normalize(fbank)
        
        return original_fbank, fbank, label_indices, datum['wav'], seg_label

    def __len__(self):
        return len(self.data)
    
    def random_uniform(self, start, end):
        val = torch.rand(1).item()
        return start + (end-start) * val

    def blur(self, fbank):
        # assert torch.min(fbank) >= 0
        F_kernel_size=int(self.random_uniform(1, self.melbins // 4))
        
        if(F_kernel_size % 2 == 0): F_kernel_size -= 1
        
        fbank = torchvision.transforms.functional.gaussian_blur(fbank, kernel_size=[F_kernel_size, F_kernel_size])
        return fbank

    def frequency_masking(self, fbank, freqm):
        bs, freq, tsteps = fbank.size()
        mask_len = int(self.random_uniform(1, freqm))
        mask_start = int(self.random_uniform(start=0, end=freq-mask_len))
        fbank[:,mask_start:mask_start+mask_len,:] *= 0.0
        # value = self.random_uniform(0.0, 1.0)
        # fbank[:,mask_start:mask_start+mask_len,:] += value
        return fbank

    def time_masking(self, fbank, timem):
        bs, freq, tsteps = fbank.size()
        mask_len = int(self.random_uniform(1, timem))
        mask_start = int(self.random_uniform(start=0, end=tsteps-mask_len))
        fbank[:,:,mask_start:mask_start+mask_len] *= 0.0
        # value = self.random_uniform(0.0, 1.0)
        # fbank[:,:,mask_start:mask_start+mask_len] += value
        return fbank