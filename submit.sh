# >>> conda initialize >>> /vol/research/dcase2022/miniconda3
__conda_setup="$('/mnt/fast/nobackup/users/hl01486/miniconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/mnt/fast/nobackup/users/hl01486/miniconda3/etc/profile.d/conda.sh" ]; then
        . "/mnt/fast/nobackup/users/hl01486/miniconda3/etc/profile.d/conda.sh"
    else
        export PATH="/mnt/fast/nobackup/users/hl01486/miniconda3/bin:$PATH"
    fi
fi

unset __conda_setup
conda activate fs2_weka

######################## ENVIRONMENT ########################
which python
rsync -r /vol/research/ai4sound/project/audio_generation/FastSpeech2 /mnt/fast/nobackup/users/hl01486/projects/audio_generation_diff
cd /mnt/fast/nobackup/users/hl01486/projects/audio_generation_diff/FastSpeech2

######################## SETUP ########################
EXP_NAME="us8k_exp6_diff_energy_pitch"

######################## RUNNING ENTRY ########################
# $1=pcen@delta_mfcc

# python3 prepare_align.py config/esc50/preprocess.yaml
# python3 preprocess.py config/esc50/preprocess.yaml

python3 train.py -p config/us8k/preprocess.yaml -m config/us8k/model.yaml -t config/us8k/train.yaml