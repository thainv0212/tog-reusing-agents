## Setup/Installation
```bash
conda env create -f env.yml
conda activate GMC3
```
Datasets in use to train multimodal encoders are available [here].(https://huggingface.co/datasets/anonymous2120/tog_anonymous/tree/main)
## - Train Model
### - Train multimodal encoder model (for all games)
```bash 
python main_rl.py -f with experiment.stage="finetune" --experiment.scenario=scenario_name #(refer to [exp_ingredients.py](gmc_code/rl/ingredients/exp_ingredients.py) for more scenarios) 
```
### - Finetune multimodal encoder model (for all games)
```bash 
python main_rl.py -f with experiment.stage="finetune" --experiment.scenario=scenario_name #(refer to [exp_ingredients.py](gmc_code/rl/ingredients/exp_ingredients.py) for more scenarios) 
```

## - Train pendulum reinforcement learning agent
### - Train policy network that use multimodal encoder's output as input
```bash 
python main_rl.py -f with experiment.stage="train_downstream_controller" --experiment.scenario=scenario_name # refer to [exp_ingredients.py](gmc_code/rl/ingredients/exp_ingredients.py) for more pendulum scenarios
```

### - Train inforcement learning agent from scratch
```bash
python main_rl.py -f with experiment.stage="train_downstream_controller" experiment.scenario_config.train_vae=1 experiment.model="sound" experiment.model_config.learner="ddpg2" experiment.scenario=scenario_name
```
### - Evaluate agent (with multimodal encoder)
```bash
python -f with experiment.stage="evaluate_downstream_controller" experiment.evaluation_mods=[1] experiment.scenario=scenario_name experiment.seed=0 # refer to [exp_ingredients.py](gmc_code/rl/ingredients/exp_ingredients.py) for more pendulum scenarios
```
### - Evaluate agent (trained from scratch)
```bash
python -f with experiment.stage="evaluate_downstream_controller" experiment.evaluation_mods=[1] experiment.model="sound" experiment.model_config.learner="ddpg2" experiment.scenario=scenario_name experiment.seed=0 # refer to [exp_ingredients.py](gmc_code/rl/ingredients/exp_ingredients.py) for more pendulum scenarios
```