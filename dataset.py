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
        np.random.shuffle(self.data)
        self.melbins = preprocess_config["preprocessing"]["mel"]["n_mel_channels"]
        self.freqm = preprocess_config["preprocessing"]["mel"]["freqm"]
        self.timem = preprocess_config["preprocessing"]["mel"]["timem"]
        self.mixup = train_config["augmentation"]["mixup"]
        
        # No augmentation during evaluation
        if(train == False):
            self.mixup = 0.0
            self.freqm = 0
            self.timem = 0
        
        self.dataset = preprocess_config['dataset']
        self.sampling_rate = preprocess_config["preprocessing"]["audio"]["sampling_rate"]
        self.segment_label_path = preprocess_config["path"]["segment_label_path"]
        self.target_length = self.preprocess_config["preprocessing"]["mel"]["target_length"]
        self.use_blur = self.preprocess_config["preprocessing"]["mel"]["blur"]
        
        try: self.label_norm = self.preprocess_config["preprocessing"]["label"]["norm"]
        except: self.label_norm=False
        
        try: self.label_top_k = self.preprocess_config["preprocessing"]["label"]["top_k"]
        except: self.label_top_k=527
        
        try: self.label_quantization = self.preprocess_config["preprocessing"]["label"]["quantization"]
        except: self.label_quantization=False
      
        try: self.label_threshold = self.preprocess_config["preprocessing"]["label"]["threshold"]
        except: self.label_threshold=False
        
        try: 
            self.label_use_original_ground_truth = self.preprocess_config["preprocessing"]["label"]["use_original_ground_truth"]
            print("Use ground truth label: %s" % self.label_use_original_ground_truth)
        except: 
            self.label_use_original_ground_truth=False
          
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
        
        self.class_reweight_matrix = np.load(preprocess_config["path"]["class_reweight_arr_path"])

    def resample(self, waveform, sr):
        if(sr==32000 and self.sampling_rate==16000):
            waveform = waveform[::2]
        return waveform

    def read_wav_file(self, filename):
        # waveform, sr = librosa.load(filename, sr=None, mono=True) # 4 times slower
        
        waveform, sr = torchaudio.load(filename) # Faster!!!
        waveform = waveform.numpy()[0,...]
        
        waveform = self.resample(waveform, sr)
        waveform = waveform - np.mean(waveform)
        return waveform
    
    def _wav2fbank(self, filename, filename2=None):
        # mixup
        if filename2 == None:
            waveform = self.read_wav_file(filename)
        # mixup
        else:
            waveform1 = self.read_wav_file(filename)
            waveform2 = self.read_wav_file(filename2)
            
            waveform1,waveform2 = waveform1[None,...], waveform2[None,...]

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

        fbank, energy = Audio.tools.get_mel_from_wav(waveform, self.STFT)
        
        # path = "/mnt/fast/nobackup/scratch4weeks/hl01486/temp_npy"
        # np.save(os.path.join(path, os.path.basename(filename)+".npy") ,np.array(fbank))
        
        fbank = torch.FloatTensor(fbank.T)
        
        n_frames = fbank.shape[0]

        p = self.target_length - n_frames
        # cut and pad
        if p > 0:
            m = torch.nn.ZeroPad2d((0, 0, 0, p))
            fbank = m(fbank)
        elif p < 0:
            fbank = fbank[0:self.target_length, :]
        
        if filename2 == None:
            return fbank, 0, waveform
        else:
            return fbank, mix_lambda, waveform
    
    def __getitem__(self, index):
        """
        returns: image, audio, nframes
        where image is a FloatTensor of size (3, H, W)
        audio is a FloatTensor of size (N_freq, N_frames) for spectrogram, or (N_frames) for waveform
        nframes is an integer
        """
        fbank, waveform, label_indices, seg_label, fname = self.feature_extraction(index)
        fbank = self.aug(fbank)
        
        if(not self.label_use_original_ground_truth): 
            seg_label = self.process_labels(seg_label)
        else:
            if(len(label_indices.shape) <= 1): 
                seg_label = label_indices[None,...]
            seg_label = np.repeat(seg_label.numpy(), 1056, 0)
            seg_label = seg_label[:self.target_length,:]
        
        return fbank, label_indices, fname, waveform, seg_label
    
    def process_labels(self, seg_label):
        # Unify the scores in the label
        if(self.label_norm): 
            seg_label = seg_label / self.class_reweight_matrix[None,...]
        # Remove the noise in the label
        seg_label[seg_label < self.label_threshold] = 0.0 
        return seg_label

    def feature_extraction(self, index):
        # Read wave file and extract feature
        if random.random() < self.mixup: 
            datum = self.data[index]
            mix_sample_idx = random.randint(0, len(self.data)-1)
            mix_datum = self.data[mix_sample_idx]
            # get the mixed fbank
            fbank, mix_lambda, waveform = self._wav2fbank(datum['wav'], mix_datum['wav'])
            # initialize the label
            label_indices = np.zeros(self.label_num)
            for label_str in datum['labels'].split(','):
                label_indices[int(self.index_dict[label_str])] += mix_lambda
            for label_str in mix_datum['labels'].split(','):
                label_indices[int(self.index_dict[label_str])] += (1.0-mix_lambda)
            
            seg_label1 = self.read_machine_label(index)
            seg_label2 = self.read_machine_label(mix_sample_idx)
            
            seg_label = mix_lambda * seg_label1 + (1-mix_lambda) * seg_label2

        else:
            datum = self.data[index]
            label_indices = np.zeros(self.label_num)
            fbank, mix_lambda, waveform = self._wav2fbank(datum['wav'])
            for label_str in datum['labels'].split(','):
                label_indices[int(self.index_dict[label_str])] = 1.0
        
            seg_label = self.read_machine_label(index)
        
        label_indices = torch.FloatTensor(label_indices)
        
        # The filename of the wav file
        fname = datum['wav']
        
        return fbank, waveform, label_indices, seg_label, fname

    def read_machine_label(self, index):
        # Read the clip-level or segment-level labels
        while(True):
            try:
                seg_label = self.read_label(self.data[index]['wav'])
                return seg_label
            except Exception as e:
                print(e)
                if(index == len(self.data)-1): index = 0
                else: index += 1
        
    def aug(self, fbank):
        assert torch.min(fbank) < 0
        fbank = fbank.exp()
        ############################### Blur and Spec Aug ####################################################
        fbank = torch.transpose(fbank, 0, 1)
        # this is just to satisfy new torchaudio version.
        fbank = fbank.unsqueeze(0)
        # self.use_blur = False
        if(self.use_blur): 
            fbank = self.blur(fbank)
        if self.freqm != 0:
            fbank = self.frequency_masking(fbank, self.freqm)
        if self.timem != 0:
            fbank = self.time_masking(fbank, self.timem) # self.timem=0
        #############################################################################################
        fbank = (fbank+1e-7).log()
        # squeeze back
        fbank = fbank.squeeze(0)
        fbank = torch.transpose(fbank, 0, 1)
        if self.noise == True:
            fbank = fbank + torch.rand(fbank.shape[0], fbank.shape[1]) * np.random.rand() / 10
            fbank = torch.roll(fbank, np.random.randint(-10, 10), 0)
        return fbank

    def read_label(self, wav_name):
        seg_label_fname = os.path.basename(wav_name).replace(".wav",".npy")
        seg_label_fpath = os.path.join(self.segment_label_path, seg_label_fname)
        seg_label = np.load(seg_label_fpath)
        
        # For the clip level label, add one more dimension
        if(len(seg_label.shape) <= 1):
            seg_label = seg_label[None,...]
        
        seg_label = np.repeat(seg_label, 1056, 0)
        seg_label = seg_label[:self.target_length,:]
        return seg_label

    def __len__(self):
        return len(self.data)
    
    def random_uniform(self, start, end):
        val = torch.rand(1).item()
        return start + (end-start) * val

    def blur(self, fbank):
        assert torch.min(fbank) >= 0
        kernel_size=int(self.random_uniform(1, self.melbins))
        fbank = torchvision.transforms.functional.gaussian_blur(fbank, kernel_size=[kernel_size, kernel_size])
        return fbank

    def frequency_masking(self, fbank, freqm):
        bs, freq, tsteps = fbank.size()
        mask_len = int(self.random_uniform(freqm // 8, freqm))
        mask_start = int(self.random_uniform(start=0, end=freq-mask_len))
        fbank[:,mask_start:mask_start+mask_len,:] *= 0.0
        return fbank

    def time_masking(self, fbank, timem):
        bs, freq, tsteps = fbank.size()
        mask_len = int(self.random_uniform(timem // 8, timem))
        mask_start = int(self.random_uniform(start=0, end=tsteps-mask_len))
        fbank[:,:,mask_start:mask_start+mask_len] *= 0.0
        return fbank