import os
import torch
import shutil

from muse.evaluation.atari.hyperhot.gmc.gmc import HyperhotGMC, HyperhotNullModel
from muse.evaluation.atari.hyperhot.vae.model import MUSE


def save_checkpoint(state,
                    is_best,
                    folder='./',
                    filename='muse_checkpoint.pth.tar'):
    torch.save(state, os.path.join(folder, filename))
    if is_best:
        shutil.copyfile(
            os.path.join(folder, filename),
            os.path.join(folder, 'best_muse_model.pth.tar'))


def load_checkpoint(checkpoint_file, use_cuda=False):
    if use_cuda:
        checkpoint = torch.load(checkpoint_file)
    else:
        checkpoint = torch.load(
            checkpoint_file, map_location=lambda storage, location: storage)

    model_config = checkpoint['model_config']
    env_config = checkpoint['env_config']
    model = make_multimodalvae(vae_config=model_config, env_config=env_config)
    model.load_state_dict(checkpoint['state_dict'])

    if use_cuda:
        model.cuda()

    return model, model_config, env_config


def make_multimodalvae(vae_config, env_config):

    model = MUSE(image_latents=vae_config['image_latent_dim'],
                  sound_latents=vae_config['sound_latent_dim'],
                  top_latents=vae_config['top_latent_dim'])
    return model

def make_gmcmodel(gmc_config):
    rep_model = gmc_config['rep_model']
    if rep_model == "gmc":
        print("Using HyperhotGMC model")
        return HyperhotGMC(name="gmc_hyperhot",
                       common_dim=gmc_config['common_dim'],
                       latent_dim=gmc_config['latent_dim'],
                       loss_type=gmc_config['loss_type'],
                       finetune=None)
    elif rep_model == "sound":
        print("Using HyperhotNullModel model")
        return HyperhotNullModel(latent_dim=gmc_config['latent_dim'])
    else:
        raise ValueError(f"Unknown rep_model: {rep_model}")

def load_gmc_checkpoint(cfg, checkpoint_file, use_cuda=True):
    print(f"Loading checkpoint from {checkpoint_file}")
    if use_cuda:
        checkpoint = torch.load(checkpoint_file)
    else:
        checkpoint = torch.load(
            checkpoint_file, map_location=lambda storage, location: storage)
    # env_config = checkpoint['env_config']
    model = make_gmcmodel(gmc_config=cfg)
    if isinstance(model, HyperhotGMC):
        info = model.load_state_dict(checkpoint['state_dict'])
        print("Load checkpoint:", info)
    else:
        print("Checkpoint is not used")
    if use_cuda:
        model.cuda()
    return model