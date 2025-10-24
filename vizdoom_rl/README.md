## Setup/Installation
```bash
conda env create -f env.yml
conda activate vizdoom
```

## Train agent
### Train with multimodal encoder model
```bash 
python train_vizdoom.py --env=env_name --train_for_env_steps=500000000 --algo=APPO --env_frameskip=4 --use_rnn=True --wide_aspect_ratio=False --num_workers=36 --num_envs_per_worker=8 --decorrelate_envs_on_one_worker=False --train_dir=train_dir --encoder_type=gmc --gmc_model_file=path_to_encoder_model --use_sound
```
### Train from scratch
```bash
python train_vizdoom.py --env=env_name --train_for_env_steps=500000000 --algo=APPO --env_frameskip=4 --use_rnn=True --num_workers=36 --num_envs_per_worker=8 --num_policies=1 --batch_size=2048 --wide_aspect_ratio=False --env=doom_basic --use_sound --audio_encoder=fft --train_dir=train_dir --encoder_conv_architecture=none
```

## Evaluate 
### Agent trained with multimodal encoder model
```bash
python enjoy_vizdoom.py --env=env --train_dir=train_dir--use_rnn=True --train_dir=train_dir --encoder_type=gmc --env_frameskip=4 --use_sound --max_num_episodes=10 --seed=19 --fps=35 --save_video --audio_encoder=fft
```
### Agent trained from scratch
```bash
python enjoy_vizdoom.py --env=env_name --env=doom_my_way_home_new_design --train_dir=train_dir--use_rnn=True --env_frameskip=4 --use_sound --max_num_episodes=10 --seed=19 --fps=35 --save_video --audio_encoder=fft
```
