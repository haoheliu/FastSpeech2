import sys
sys.path.append("..")

from evaluation.datasets.load_mel import MelDataset,load_npy_data
from evaluation.metrics.ndb import *
import numpy as np
import argparse
import torch
from torch.utils.data import DataLoader
from evaluation.feature_extractors.melception import Melception
from tqdm import tqdm
from evaluation.metrics import gs
from evaluation.metrics.fid import calculate_fid
from evaluation.metrics.isc import calculate_isc
from evaluation.metrics.kid import calculate_kid
from evaluation.metrics.kl import calculate_kl

import audio as Audio

class EvaluationHelper():
    def __init__(self, preprocess_config, model_config, train_config, device) -> None:
        
        self.preprocess_config=preprocess_config
        self.model_config=model_config
        self.train_config=train_config
        self.device = device
        
        self._stft = Audio.stft.TacotronSTFT(
            preprocess_config["preprocessing"]["stft"]["filter_length"],
            preprocess_config["preprocessing"]["stft"]["hop_length"],
            preprocess_config["preprocessing"]["stft"]["win_length"],
            preprocess_config["preprocessing"]["mel"]["n_mel_channels"],
            preprocess_config["preprocessing"]["audio"]["sampling_rate"],
            preprocess_config["preprocessing"]["mel"]["mel_fmin"],
            preprocess_config["preprocessing"]["mel"]["mel_fmax"],
        )
        
        self.mel_model=Melception(num_classes=309,features_list=['logits_unbiased','2048','logits'],feature_extractor_weights_path="/mnt/fast/nobackup/users/hl01486/projects/general_audio_generation/conditional_transfer/FastSpeech2/evaluation/logs/melception/melception.pt")
        self.mel_model.eval()
        self.mel_model.to(self.device)
    
        self.fbin_mean, self.fbin_std = None, None
        
    def calculate_stats_for_normalization(self, path, max_num=1000):
        num_workers=0
        dataset = DataLoader(MelDataset(path, 
                                        self._stft, 
                                        self.preprocess_config["preprocessing"]["audio"]["sampling_rate"],
                                        self.fbin_mean,
                                        self.fbin_std), 
                             batch_size=1, 
                             sampler=None, 
                             num_workers=num_workers)
        
        mean,std = [],[]
        
        for id, (batch,filename) in tqdm(enumerate(dataset)):
            batch = batch.float().numpy() # [1, 64, 1001]
            mean.append(np.mean(batch, axis=2))
            std.append(np.std(batch, axis=2))
            if(max_num is not None and id > max_num):
                break
        
        mean = np.concatenate(mean, axis=0)
        std = np.concatenate(std, axis=0)
        
        self.fbin_mean = np.mean(mean, axis=0)
        self.fbin_std = np.mean(std, axis=0)
            
    def main(self, o_filepath,resultpath,same_name=False,number_of_bins=10,evaluation_num=10,cache_folder='./results/mnist_toy_example_ndb_cache',iter_num=40):

        # Use the ground truth audio file to calculate mean and std
        self.calculate_stats_for_normalization(resultpath)

        gsm = self.getgsmscore(o_filepath, resultpath, iter_num)

        ndb = self.getndbscore(o_filepath, resultpath, number_of_bins, evaluation_num, cache_folder)

        metrics=self.calculate_metrics(o_filepath,resultpath,same_name)
        
        return gsm, ndb, metrics

    def getndbscore(self, output, result,number_of_bins=30,evaluation_num=50,cache_folder='./results/mnist_toy_example_ndb_cache'):
        print("calculating the ndb score:")
        num_workers=0
        
        outputloader = DataLoader(MelDataset(output, self._stft, self.preprocess_config["preprocessing"]["audio"]["sampling_rate"], self.fbin_mean, self.fbin_std, augment=True), batch_size=1, sampler=None, num_workers=num_workers)
        resultloader = DataLoader(MelDataset(result, self._stft, self.preprocess_config["preprocessing"]["audio"]["sampling_rate"], self.fbin_mean, self.fbin_std), batch_size=1, sampler=None, num_workers=num_workers)
        
        n_query = evaluation_num
        train_samples=load_npy_data(outputloader)

        #print('Initialize NDB bins with training samples')
        mnist_ndb = NDB(training_data=train_samples, number_of_bins=number_of_bins, z_threshold=None, whitening=False,
                        cache_folder=cache_folder)
        
        result_samples = load_npy_data(resultloader)
        results=mnist_ndb.evaluate(self.sample_from(result_samples, n_query), 'generated result')
        plt.figure()
        mnist_ndb.plot_results()

    def getgsmscore(self, output, result, iter_num=40):
        num_workers=0
        
        print("calculating the gsm score:")
        
        outputloader = DataLoader(MelDataset(output, self._stft, self.preprocess_config["preprocessing"]["audio"]["sampling_rate"], self.fbin_mean, self.fbin_std, augment=True), batch_size=1, sampler=None, num_workers=num_workers)
        resultloader = DataLoader(MelDataset(result, self._stft, self.preprocess_config["preprocessing"]["audio"]["sampling_rate"], self.fbin_mean, self.fbin_std), batch_size=1, sampler=None, num_workers=num_workers)
        
        x_train = load_npy_data(outputloader)
        
        x_1 = x_train
        newshape=int(x_1.shape[1]/8)
        x_1 = np.reshape(x_1, (-1, newshape))
        rlts = gs.rlts(x_1, gamma=1.0 / 128, n=iter_num)
        mrlt = np.mean(rlts, axis=0)

        gs.fancy_plot(mrlt, label='MRLT of data_1',color="C0")
        plt.xlim([0, 30])
        plt.legend()

        x_train= load_npy_data(resultloader)

        x_1 = x_train
        x_1 = np.reshape(x_1, (-1, newshape))
        rlts = gs.rlts(x_1, gamma=1.0 / 128, n=iter_num)

        mrlt = np.mean(rlts, axis=0)

        gs.fancy_plot(mrlt, label='MRLT of data_2',color="orange")
        plt.xlim([0, 30])
        plt.legend()
        plt.show()

    def calculate_metrics(self, output, result, same_name):
        torch.manual_seed(0)
        num_workers=0
        
        outputloader = DataLoader(MelDataset(output, self._stft, self.preprocess_config["preprocessing"]["audio"]["sampling_rate"], self.fbin_mean, self.fbin_std, augment=True), batch_size=1, sampler=None, num_workers=num_workers)
        resultloader = DataLoader(MelDataset(result, self._stft, self.preprocess_config["preprocessing"]["audio"]["sampling_rate"], self.fbin_mean, self.fbin_std), batch_size=1, sampler=None, num_workers=num_workers)

        out = {}

        print('Extracting features from input_1')
        featuresdict_1 = self.get_featuresdict(outputloader)
        print('Extracting features from input_2')
        featuresdict_2 = self.get_featuresdict(resultloader)

        # if cfg.have_kl:
        metric_kl = calculate_kl(featuresdict_1, featuresdict_2, "logits",same_name)
        out.update(metric_kl)
        # if cfg.have_isc:
        metric_isc = calculate_isc(featuresdict_1, feat_layer_name="logits_unbiased", splits=4, samples_shuffle= True, rng_seed=2020)
        out.update(metric_isc)
        # if cfg.have_fid:
        metric_fid = calculate_fid(featuresdict_1, featuresdict_2, feat_layer_name="2048")
        out.update(metric_fid)
        # if cfg.have_kid:
        metric_kid = calculate_kid(featuresdict_1, featuresdict_2, feat_layer_name="2048", subsets=100, subset_size=1000, degree=3, gamma=None, coef0=1, rng_seed=2020)
        out.update(metric_kid)

        print('\n'.join((f'{k}: {v:.7f}' for k, v in out.items())))
        print("\n")
        print(
            f'KL: {out.get("kullback_leibler_divergence", float("nan")):8.5f};',
            f'ISc: {out.get("inception_score_mean", float("nan")):8.5f} ({out.get("inception_score_std", float("nan")):5f});',
            f'FID: {out.get("frechet_inception_distance", float("nan")):8.5f};',
            f'KID: {out.get("kernel_inception_distance_mean", float("nan")):.5f}',
            f'({out.get("kernel_inception_distance_std", float("nan")):.5f})'
        )

    def get_featuresdict(self, dataloader):

        out = None
        out_meta = None

        # transforms=StandardNormalizeAudio()

        for batch,filename in tqdm(dataloader):
            metadict = {
                'file_path_': filename,
            }

            # batch = transforms(batch) 
            batch = batch.float().to(self.device)

            with torch.no_grad():
                features = self.mel_model(batch)

            featuresdict = self.mel_model.convert_features_tuple_to_dict(features)
            featuresdict = {k: [v.cpu()] for k, v in featuresdict.items()}

            if out is None:
                out = featuresdict
            else:
                out = {k: out[k] + featuresdict[k] for k in out.keys()}

            if out_meta is None:
                out_meta = metadict
            else:
                out_meta = {k: out_meta[k] + metadict[k] for k in out_meta.keys()}

        out = {k: torch.cat(v, dim=0) for k, v in out.items()}
        return {**out, **out_meta}


    def sample_from(self, samples, number_to_use):
        assert samples.shape[0] >= number_to_use
        rand_order = np.random.permutation(samples.shape[0])
        return samples[rand_order[:samples.shape[0]], :]

if __name__ == '__main__':
    import yaml
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--restore_step", type=int, required=False)
    parser.add_argument(
        "-p",
        "--preprocess_config",
        type=str,
        required=False,
        help="path to preprocess.yaml",
        default="config/test/test/preprocess.yaml"
    )
    parser.add_argument(
        "-m", "--model_config", type=str, required=False, help="path to model.yaml", default="config/test/test/model.yaml"
    )
    parser.add_argument(
        "-t", "--train_config", type=str, required=False, help="path to train.yaml", default="config/test/test/train.yaml"
    )
    args = parser.parse_args()

    # Read Config
    preprocess_config = yaml.load(
        open(args.preprocess_config, "r"), Loader=yaml.FullLoader
    )
    model_config = yaml.load(open(args.model_config, "r"), Loader=yaml.FullLoader)
    train_config = yaml.load(open(args.train_config, "r"), Loader=yaml.FullLoader)
    
    o_filepath = "/mnt/fast/nobackup/users/hl01486/projects/general_audio_generation/conditional_transfer/FastSpeech2/data/testdata/unpaired/60w/target"
    # resultpath = "/mnt/fast/nobackup/users/hl01486/projects/general_audio_generation/conditional_transfer/FastSpeech2/data/testdata/unpaired/60w/target"
    resultpath = "/mnt/fast/nobackup/users/hl01486/projects/general_audio_generation/conditional_transfer/FastSpeech2/data/testdata/unpaired/60w/synthesis"
    
    same_name = False
    
    config=(preprocess_config, model_config, train_config)
    device = torch.device(f'cuda:{0}')
    
    evaluator = EvaluationHelper(preprocess_config, model_config, train_config, device)
    
    evaluator.main(o_filepath,resultpath,same_name)
    # evaluator.calculate_stats_for_normalization()