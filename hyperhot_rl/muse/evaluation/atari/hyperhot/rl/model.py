import torch.nn as nn
from muse.evaluation.atari.hyperhot.gmc.gmc import HyperhotSoundOnly

class Network(nn.Module):
    def __init__(self, n_states, n_actions, layers_sizes):
        super(Network, self).__init__()

        layers = []
        pre = n_states
        for ls in layers_sizes:
            pos = ls
            ll = nn.Linear(pre, pos)
            nn.init.xavier_uniform_(ll.weight)
            layers.append(ll)
            layers.append(nn.ReLU())

            pre = pos
        layers.append(nn.Linear(pre, n_actions))

        self.linear = nn.Sequential(*layers)

    def forward(self, x):
        return self.linear(x.view(x.size(0), -1))


class DQN(nn.Module):
    def __init__(self, n_states, n_actions, layers_sizes, cuda):
        super(DQN, self).__init__()

        self.n_states = n_states
        self.n_actions = n_actions

        device = 'cuda' if cuda else 'cpu'

        self.net = Network(n_states, n_actions, layers_sizes).to(device)
        self.target = Network(n_states, n_actions, layers_sizes).to(device)
        self.target.load_state_dict(self.net.state_dict())
        self.target.eval()

class Network2(nn.Module):
    def __init__(self, n_states, n_actions, layers_sizes):
        super(Network2, self).__init__()
        self.encoder = HyperhotSoundOnly(None, 64, n_states)
        layers = []
        pre = n_states
        for ls in layers_sizes:
            pos = ls
            ll = nn.Linear(pre, pos)
            nn.init.xavier_uniform_(ll.weight)
            layers.append(ll)
            layers.append(nn.ReLU())

            pre = pos
        layers.append(nn.Linear(pre, n_actions))

        self.linear = nn.Sequential(*layers)

    def forward(self, x):
        return self.linear(self.encoder(x.view(x.size(0), -1))[0])

class DQN2(nn.Module):
    def __init__(self, n_states, n_actions, layers_sizes, cuda):
        super(DQN2, self).__init__()

        self.n_states = n_states
        self.n_actions = n_actions

        device = 'cuda' if cuda else 'cpu'

        self.net = Network2(n_states, n_actions, layers_sizes).to(device)
        self.target = Network2(n_states, n_actions, layers_sizes).to(device)
        self.target.load_state_dict(self.net.state_dict())
        self.target.eval()

class DQN3(DQN2):
    def __init__(self, n_states, n_actions, layers_sizes, cuda, model_file=None, finetune_encoder=False):
        super(DQN2, self).__init__()

        self.n_states = n_states
        self.n_actions = n_actions

        device = 'cuda' if cuda else 'cpu'

        self.net = Network2(n_states, n_actions, layers_sizes).to(device)
        self.target = Network2(n_states, n_actions, layers_sizes).to(device)
        self.target.load_state_dict(self.net.state_dict())
        self.target.eval()

    def train(self, mode: bool = True):
        # self.eval()
        # for p in self.children():
        #     p.eval()
        super().train(False)
        freeze_module(self)
        unfreeze_module(self.net.encoder)
        self.net.encoder.train(mode)
        return self

def freeze_module(module):
    for param in module.parameters():
        param.requires_grad = False

def unfreeze_module(module):
    for param in module.parameters():
        param.requires_grad = True
