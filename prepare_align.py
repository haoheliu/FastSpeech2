import argparse

import yaml

from preprocessor import ljspeech, aishell3, libritts, esc50, us8k


def main(config):
    if "LJSpeech" in config["dataset"]:
        ljspeech.prepare_align(config)
    if "AISHELL3" in config["dataset"]:
        aishell3.prepare_align(config)
    if "LibriTTS" in config["dataset"]:
        libritts.prepare_align(config)
    if "esc50" in config["dataset"]:
        esc50.prepare_align(config)
    if "us8k" in config["dataset"]:
        us8k.prepare_align(config)
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str, help="path to preprocess.yaml")
    args = parser.parse_args()

    config = yaml.load(open(args.config, "r"), Loader=yaml.FullLoader)
    main(config)
