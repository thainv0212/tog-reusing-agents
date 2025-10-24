## Setup/Installation
```bash
conda env create -f env.yml
conda activate muse39
```

## Train Agent
### Train from scratch
```bash 
python train_rl.py -f with gmc.env=env_name gmc.train_config.rep_model="sound" gmc.train_config.learner="dqn2"
```
### Train with multimodal encoder
```bash 
python train_rl.py -f with gmc.env=env_name gmc.train_config.rep_model="gmc" gmc.train_config.learner="gmc"
```

## Evaluate agent
### Agent trained from scratch
```bash 
python eval_pipeline.py -f with gmc.train_config.learner="dqn2" gmc.env=env
```

### Agent that uses multimodal encoder
```bash 
python eval_pipeline.py -f with gmc.train_config.learner="dqn" gmc.env=env
```