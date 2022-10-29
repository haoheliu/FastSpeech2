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
    def __init__(self, preprocess_config, train_config, train=True):
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
        self.melbins = preprocess_config["preprocessing"]["mel"]["n_mel_channels"]
        self.freqm = preprocess_config["preprocessing"]["mel"]["freqm"]
        self.timem = preprocess_config["preprocessing"]["mel"]["timem"]
        self.mixup = train_config["augmentation"]["mixup"]
        self.dataset = preprocess_config['dataset']
        self.sampling_rate = preprocess_config["preprocessing"]["audio"]["sampling_rate"]
        
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

    def resample(self, waveform, sr):
        if(sr==32000 and self.sampling_rate==16000):
            waveform = waveform[::2]
        return waveform

    def _wav2fbank(self, filename, filename2=None):
        # mixup
        if filename2 == None:
            waveform, sr = librosa.load(filename, sr=None, mono=True)
            waveform = self.resample(waveform, sr)
            # waveform = waveform - np.mean(waveform)
        # mixup
        else:
            waveform1, sr = librosa.load(filename, sr=None, mono=True)
            waveform1 = self.resample(waveform1, sr)
            waveform2, sr2 = librosa.load(filename2, sr=None, mono=True)
            waveform2 = self.resample(waveform2, sr2)
            
            waveform1,waveform2 = waveform1[None,...], waveform2[None,...]
            waveform1 = waveform1 - np.mean(waveform1)
            waveform2 = waveform2 - np.mean(waveform2)

            if waveform1.shape[1] != waveform2.shape[1]:
                if waveform1.shape[1] > waveform2.shape[1]:
                    # padding
                    temp_wav = np.zeros((1, waveform1.shape[1]))
                    temp_wav[0, 0:waveform2.shape[1]] = waveform2
                    waveform2 = temp_wav
                else:
                    # cutting
                    waveform2 = waveform2[0, 0:waveform1.shape[1]]

            # sample lambda from uniform distribution
            #mix_lambda = random.random()
            # sample lambda from beta distribtion
            mix_lambda = np.random.beta(10, 10)

            mix_waveform = mix_lambda * waveform1 + (1 - mix_lambda) * waveform2
            waveform = mix_waveform - np.mean(mix_waveform)
            waveform=waveform[0,...]

        # waveform = waveform / np.max(np.abs(waveform))
        # fbank = torchaudio.compliance.kaldi.fbank(waveform, htk_compat=True, sample_frequency=sr, use_energy=False,
        #                                           window_type='hanning', num_mel_bins=self.melbins, dither=0.0, frame_shift=10)
        fbank, energy = Audio.tools.get_mel_from_wav(waveform, self.STFT)
        fbank = torch.FloatTensor(fbank.T)
        
        target_length = self.preprocess_config["preprocessing"]["mel"]["target_length"]
        n_frames = fbank.shape[0]

        p = target_length - n_frames
        # cut and pad
        if p > 0:
            m = torch.nn.ZeroPad2d((0, 0, 0, p))
            fbank = m(fbank)
        elif p < 0:
            fbank = fbank[0:target_length, :]
            
        # fbank = fbank.permute
        
        if filename2 == None:
            return fbank, 0
        else:
            return fbank, mix_lambda

    def __getitem__(self, index):
        """
        returns: image, audio, nframes
        where image is a FloatTensor of size (3, H, W)
        audio is a FloatTensor of size (N_freq, N_frames) for spectrogram, or (N_frames) for waveform
        nframes is an integer
        """
        # do mix-up for this sample (controlled by the given mixup rate)
        if random.random() < self.mixup:
            datum = self.data[index]
            # find another sample to mix, also do balance sampling
            # sample the other sample from the multinomial distribution, will make the performance worse
            # mix_sample_idx = np.random.choice(len(self.data), p=self.sample_weight_file)
            # sample the other sample from the uniform distribution
            mix_sample_idx = random.randint(0, len(self.data)-1)
            mix_datum = self.data[mix_sample_idx]
            # get the mixed fbank
            fbank, mix_lambda = self._wav2fbank(datum['wav'], mix_datum['wav'])
            # initialize the label
            label_indices = np.zeros(self.label_num)
            # add sample 1 labels
            for label_str in datum['labels'].split(','):
                label_indices[int(self.index_dict[label_str])] += mix_lambda
            # add sample 2 labels
            for label_str in mix_datum['labels'].split(','):
                label_indices[int(self.index_dict[label_str])] += (1.0-mix_lambda)
            label_indices = torch.FloatTensor(label_indices)
        # if not do mixup
        else:
            datum = self.data[index]
            label_indices = np.zeros(self.label_num)
            fbank, mix_lambda = self._wav2fbank(datum['wav'])
            for label_str in datum['labels'].split(','):
                label_indices[int(self.index_dict[label_str])] = 1.0

            label_indices = torch.FloatTensor(label_indices)

        # SpecAug, not do for eval set
        freqm = torchaudio.transforms.FrequencyMasking(self.freqm)
        timem = torchaudio.transforms.TimeMasking(self.timem)
        fbank = torch.transpose(fbank, 0, 1)
        # this is just to satisfy new torchaudio version.
        fbank = fbank.unsqueeze(0)
        if self.freqm != 0:
            fbank = freqm(fbank)
        if self.timem != 0:
            fbank = timem(fbank)
        # squeeze back
        fbank = fbank.squeeze(0)
        fbank = torch.transpose(fbank, 0, 1)

        # # normalize the input for both training and test
        # if not self.skip_norm:
        #     fbank = (fbank - self.norm_mean) / (self.norm_std)
        # # skip normalization the input if you are trying to get the normalization stats.
        # else:
        #     pass

        if self.noise == True:
            fbank = fbank + torch.rand(fbank.shape[0], fbank.shape[1]) * np.random.rand() / 10
            fbank = torch.roll(fbank, np.random.randint(-10, 10), 0)

        # the output fbank shape is [time_frame_num, frequency_bins], e.g., [1024, 128]
        return fbank, label_indices

    def __len__(self):
        return len(self.data)
    