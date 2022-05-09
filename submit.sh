# >>> conda initialize >>>
__conda_setup="$('/vol/research/dcase2022/miniconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/vol/research/dcase2022/miniconda3/etc/profile.d/conda.sh" ]; then
        . "/vol/research/dcase2022/miniconda3/etc/profile.d/conda.sh"
    else
        export PATH="/vol/research/dcase2022/miniconda3/bin:$PATH"
    fi
fi
unset __conda_setup
conda activate fs2
######################## ENVIRONMENT ########################
which python
cd /vol/research/ai4sound/project/audio_generation/FastSpeech2

######################## SETUP ########################
EXP_NAME="fs2_esc50_16k_gen"

######################## RUNNING ENTRY ########################
# $1=pcen@delta_mfcc

python3 train.py -p config/esc50/preprocess.yaml -m config/esc50/model.yaml -t config/esc50/train.yaml