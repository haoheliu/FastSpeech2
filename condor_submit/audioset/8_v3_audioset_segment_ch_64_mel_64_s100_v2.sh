IDENTIFIER=29_10_2022_19_30

AUTOGEN=1
######################## ENVIRONMENT ########################
eval "$('/mnt/fast/nobackup/users/hl01486/miniconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
conda activate psla
which python

######################## SETUP ########################
DATA="/mnt/fast/nobackup/scratch4weeks/hl01486/datasets/2million_audioset_wav&/mnt/fast/nobackup/scratch4weeks/hl01486/datasets/2million_audioset_wav"
LOG="/mnt/fast/nobackup/users/hl01486/projects/general_audio_generation/conditional_transfer/FastSpeech2&/mnt/fast/nobackup/scratch4weeks/hl01486/project/tmp/fs2_vggsound_$IDENTIFIER"
PROJECT="/mnt/fast/nobackup/users/hl01486/projects/general_audio_generation/conditional_transfer/FastSpeech2&/mnt/fast/nobackup/scratch4weeks/hl01486/project/tmp/fs2_vggsound_$IDENTIFIER"

######################## RUNNING ENTRY ########################
cd /mnt/fast/nobackup/scratch4weeks/hl01486/project/tmp/fs2_vggsound_$IDENTIFIER

python3 train.py -p config/v3_audioset_segment_ch64_mel64_S100/preprocess.yaml -m config/v3_audioset_segment_ch64_mel64_S100/modelv2.yaml -t config/v3_audioset_segment_ch64_mel64_S100/trainv2.yaml