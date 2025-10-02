import os
import torch
import shutil
from muse.evaluation.atari.hyperhot.rl.model import DQN, DQN2, DQN3


def save_checkpoint(state,
                    is_best,
                    folder='./',
                    filename='dqn_checkpoint.pth.tar',
                    best_filename='best_dqn_model.pth.tar'):
    torch.save(state, os.path.join(folder, filename))
    if is_best:
        shutil.copyfile(
            os.path.join(folder, filename), os.path.join(
                folder, best_filename))


def load_checkpoint(checkpoint_file, use_cuda=False, learner='dqn'):
    if use_cuda:
        checkpoint = torch.load(checkpoint_file)
    else:
        checkpoint = torch.load(checkpoint_file, map_location='cpu')

    dqn_config = checkpoint['dqn_config']
    env_config = checkpoint['env_config']
    if learner == 'dqn':
        dqn = DQN(env_config['n_states'], env_config['n_actions'],
              dqn_config['layers_sizes'], use_cuda)
    elif learner == 'dqn2':
        dqn = DQN2(env_config['n_states'], env_config['n_actions'],dqn_config['layers_sizes'], use_cuda)
    elif learner == 'dqn3':
        dqn = DQN3(env_config['n_states'], env_config['n_actions'],dqn_config['layers_sizes'], use_cuda)
    else:
        raise ValueError('Unknown learner type: {}'.format(learner))
    info = dqn.load_state_dict(checkpoint['state_dict'])
    print(info)

    return dqn, dqn_config