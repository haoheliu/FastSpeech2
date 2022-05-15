import os

import librosa
import numpy as np
from scipy.io import wavfile
from tqdm import tqdm

from text import _clean_text
import glob

embeddings_path = "/vol/research/ai4sound/project/audio_generation/FastSpeech2/raw_data/embeddings"

def prepare_align(config):
    in_dir = config["path"]["corpus_path"]
    out_dir = config["path"]["raw_path"]
    sampling_rate = config["preprocessing"]["audio"]["sampling_rate"]
    max_wav_value = config["preprocessing"]["audio"]["max_wav_value"]
    cleaners = config["preprocessing"]["text"]["text_cleaners"]
    
    path = glob.glob(os.path.join(in_dir,"*/*.wav"))
    for wav_path in tqdm(path):
        if os.path.exists(wav_path):
            speaker = os.path.basename(os.path.dirname(wav_path))
            os.makedirs(os.path.join(out_dir, speaker), exist_ok=True)
            wav, _ = librosa.load(wav_path, sampling_rate)
            wav = wav / max(abs(wav)) * max_wav_value
            wavfile.write(
                os.path.join(out_dir, speaker, os.path.basename(wav_path)),
                sampling_rate,
                wav.astype(np.int16),
            )
            cmd = "cp %s %s" % (os.path.join(embeddings_path, os.path.basename(wav_path).replace(".wav",".npy")), \
                os.path.join(out_dir, speaker))
            os.system(cmd)