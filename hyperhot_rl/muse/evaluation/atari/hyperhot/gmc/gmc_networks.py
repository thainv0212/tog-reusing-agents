import torch
import torch.nn as nn
import torch.nn.functional as F


# Pendulum
class PendulumCommonEncoder(nn.Module):
    def __init__(self, common_dim, latent_dim):
        super(PendulumCommonEncoder, self).__init__()
        # Variables
        self.common_dim = common_dim
        self.latent_dim = latent_dim

        self.feature_extractor = nn.Sequential(
            nn.Linear(common_dim, 128), Swish(), nn.Linear(128, latent_dim),
        )

    def forward(self, x):
        return F.normalize(self.feature_extractor(x), dim=-1)


class PendulumImageProcessor(nn.Module):
    def __init__(self, common_dim):
        super(PendulumImageProcessor, self).__init__()
        self.common_dim = common_dim

        self.image_features = nn.Sequential(
            nn.Conv2d(2, 32, 4, 2, 1, bias=False),
            Swish(),
            nn.Conv2d(32, 64, 4, 2, 1, bias=False),
            Swish(),
        )

        self.projector = nn.Linear(14400, common_dim)

    def forward(self, x):
        x = self.image_features(x)
        x = x.view(x.size(0), -1)
        return self.projector(x)


class PendulumSoundProcessor(nn.Module):
    def __init__(self, common_dim):
        super(PendulumSoundProcessor, self).__init__()

        self.common_dim = common_dim
        self.n_stack = 2
        self.sound_channels = 3
        self.sound_length = 2
        self.unrolled_sound_input = (
                self.n_stack * self.sound_channels * self.sound_length
        )

        self.snd_features = nn.Sequential(
            nn.Linear(self.unrolled_sound_input, 50),
            Swish(),
            nn.Linear(50, 50),
            Swish(),
        )

        self.projector = nn.Linear(50, common_dim)

    def forward(self, x):
        x = x.view(-1, self.unrolled_sound_input)
        h = self.snd_features(x)
        return self.projector(h)


# Pendulum
class PendulumJointProcessor(nn.Module):
    def __init__(self, common_dim, finetune=None):
        super(PendulumJointProcessor, self).__init__()
        # Variables
        self.common_dim = common_dim
        self.n_stack = 2
        self.sound_channels = 3
        self.sound_length = 2
        self.unrolled_sound_input = (
                self.n_stack * self.sound_channels * self.sound_length
        )

        self.img_features = nn.Sequential(
            nn.Conv2d(2, 32, 4, 2, 1, bias=False),
            Swish(),
            nn.Conv2d(32, 64, 4, 2, 1, bias=False),
            Swish(),
        )

        self.snd_features = nn.Sequential(
            nn.Linear(self.unrolled_sound_input, 50),
            Swish(),
            nn.Linear(50, 50),
            Swish(),
        )

        self.projector = nn.Linear(14400 + 50, common_dim)
        if finetune is not None:
            if finetune != "image":
                self.img_features.eval()
                freeze_module(self.img_features)
            if finetune != "sound":
                self.snd_features.eval()
                freeze_module(self.snd_features)
            self.projector.eval()
            freeze_module(self.projector)

    def forward(self, x):

        x_img, x_snd = x[0], x[1]

        x_img = self.img_features(x_img)
        x_img = x_img.view(x_img.size(0), -1)

        x_snd = x_snd.view(-1, self.unrolled_sound_input)
        x_snd = self.snd_features(x_snd)
        return self.projector(torch.cat((x_img, x_snd), dim=-1))


class HyperhotImageProcessor(nn.Module):
    """Parametrizes q(z|x).
    @param n_latents: integer
                      number of latent dimensions
    """

    def __init__(self, common_dim):
        super(HyperhotImageProcessor, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(2, 32, 8, 4, 2, bias=False),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1, 0, bias=False),
            nn.ReLU(),
        )

        self.classifier = nn.Sequential(
            nn.Linear(4096, 512),
            nn.ReLU())

        self.latent_dim = common_dim

        self.fc_mu = nn.Linear(512, common_dim)
        # self.fc_logvar = nn.Linear(512, latent_dim)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return self.fc_mu(x)  # , self.fc_logvar(x)


class HyperhotSoundProcessor(nn.Module):
    def __init__(self, common_dim):
        super(HyperhotSoundProcessor, self).__init__()

        self.n_stack = 2
        self.sound_channels = 4
        self.sound_length = 1047

        self.unrolled_sound_input = self.n_stack * self.sound_channels * self.sound_length

        self.fc1 = nn.Linear(self.unrolled_sound_input, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 512)
        self.bn2 = nn.BatchNorm1d(512)
        self.fc31 = nn.Linear(512, common_dim)
        # self.fc32 = nn.Linear(512, latent_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.view(-1, self.unrolled_sound_input)
        h = self.relu(self.bn1(self.fc1(x)))
        h = self.relu(self.bn2(self.fc2(h)))
        return self.fc31(h)  # , self.fc32(h)


class SafeBatchNorm1d(nn.BatchNorm1d):
    def forward(self, x):
        n_vals = x.size(0) if x.dim() == 2 else x.size(0) * x.size(2)
        if self.training and n_vals == 1:
            # use running stats; do NOT update them
            return F.batch_norm(
                x, self.running_mean, self.running_var,
                self.weight, self.bias,
                training=False, momentum=0.0, eps=self.eps
            )
        return super().forward(x)


class HyperhotSoundProcessor2(nn.Module):
    def __init__(self, common_dim):
        super(HyperhotSoundProcessor2, self).__init__()

        self.n_stack = 2
        self.sound_channels = 4
        self.sound_length = 1047

        self.unrolled_sound_input = self.n_stack * self.sound_channels * self.sound_length

        self.fc1 = nn.Linear(self.unrolled_sound_input, 512)
        self.bn1 = SafeBatchNorm1d(512)
        self.fc2 = nn.Linear(512, 512)
        self.bn2 = SafeBatchNorm1d(512)
        self.fc31 = nn.Linear(512, common_dim)
        # self.fc32 = nn.Linear(512, latent_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.view(-1, self.unrolled_sound_input)
        # h = self.relu(self.fc1(x))
        # h = self.relu(self.fc2(h))
        h = self.relu(self.bn1(self.fc1(x)))
        h = self.relu(self.bn2(self.fc2(h)))
        return self.fc31(h)  # , self.fc32(h)


class HyperhotCommonEncoder(nn.Module):
    def __init__(self, common_dim, latent_dim):
        super(HyperhotCommonEncoder, self).__init__()
        # Variables
        self.common_dim = common_dim
        self.latent_dim = latent_dim

        self.feature_extractor = nn.Sequential(
            nn.Linear(common_dim, 128), Swish(), nn.Linear(128, latent_dim),
        )

    def forward(self, x):
        return F.normalize(self.feature_extractor(x), dim=-1)


class HyperhotJointProcessor(nn.Module):
    def __init__(self, common_dim, finetune=None):
        super(HyperhotJointProcessor, self).__init__()
        # Variables
        self.common_dim = common_dim
        # self.n_stack = 2
        # self.sound_channels = 3
        # self.sound_length = 2
        self.n_stack = 2
        self.sound_channels = 4
        self.sound_length = 1047
        self.unrolled_sound_input = (
                self.n_stack * self.sound_channels * self.sound_length
        )

        self.img_features = nn.Sequential(
            nn.Conv2d(2, 32, 8, 4, 2, bias=False),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1, 0, bias=False),
            nn.ReLU(),
        )

        self.snd_features = nn.Sequential(
            nn.Linear(self.unrolled_sound_input, 512),
            nn.BatchNorm1d(512),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
        )

        self.projector = nn.Linear(4096 + 512, common_dim)
        if finetune is not None:
            if finetune != "image":
                self.img_features.eval()
                freeze_module(self.img_features)
            if finetune != "sound":
                self.snd_features.eval()
                freeze_module(self.snd_features)
            self.projector.eval()
            freeze_module(self.projector)

    def forward(self, x):

        x_img, x_snd = x[0], x[1]

        x_img = self.img_features(x_img)
        x_img = x_img.view(x_img.size(0), -1)

        x_snd = x_snd.view(-1, self.unrolled_sound_input)
        x_snd = self.snd_features(x_snd)
        return self.projector(torch.cat((x_img, x_snd), dim=-1))


"""


Extra components


"""


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


def freeze_module(module):
    for param in module.parameters():
        param.requires_grad = False
