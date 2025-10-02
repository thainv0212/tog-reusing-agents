import sacred

#############
# HyperHot #
#############

hyperhot_ingredient = sacred.Ingredient('hyperhot')


@hyperhot_ingredient.config
def hyperhot_config():
    train_dataset_samples = 32000
    test_dataset_samples = 8000
    image_side = 80
    sound_samples = 1047
    n_stack = 2
    n_enemies = 4
    pacifist_mode = False
    time_limit = 15
    sound_receivers = ["LEFT_BOTTOM", "RIGHT_BOTTOM", "LEFT_SHIP", "RIGHT_SHIP"]


#############
# VAE       #
#############

vae_ingredient = sacred.Ingredient('vae')


@vae_ingredient.config
def vae_config():
    # Training Hyperparameters
    epochs = 250
    batch_size = 128
    learning_rate = 1e-3

    # Capacity parameters
    lambda_image = 0.02
    lambda_sound = 0.015
    beta_image = 0.00001
    beta_sound = 0.00001
    gamma_image = 0.02
    gamma_sound = 0.015
    beta_top = 0.00001
    alpha_fpa = 0.00001

    wup_mod_epochs = 0
    wup_top_epochs = 0

    # Model Parameters
    image_latent_dim = 64
    sound_latent_dim = 64
    top_latent_dim = 40

    seed = 4


vae_debug_ingredient = sacred.Ingredient('vae_debug')


@vae_debug_ingredient.config
def debug_config():
    artifact_storage_interval = 50


#######
# DQN #
#######

dqn_ingredient = sacred.Ingredient('dqn')


@dqn_ingredient.config
def dqn_config():
    # max number of training frames
    max_frames = 2000000
    # max_frames = 30000
    # discount factor
    gamma = 0.99
    # replay buffer size
    memory_size = 250000

    # sizes of linear layers
    layers_sizes = [512]

    batch_size = 128
    learning_rate = 1e-4

    target_network_update_freq = 10000
    policy_network_update_freq = 4

    # Policy params
    eps_initial = 0.1
    eps_end = 0.01
    # number of frames for eps to decrease from initial to end
    eps_decay = 1500000
    # number of frames where we should execute a random policy
    replay_buffer_start_size = 50000

    condition_on_joint = True
    condition_on_image = False
    condition_on_sound = False


dqn_eval_ingredient = sacred.Ingredient('dqn_eval')


@dqn_eval_ingredient.config
def dqn_eval_config():
    # evaluate every x episodes
    eval_frequency = 150
    # number of eval episodes (each time)
    eval_length = 50


dqn_setup_vae_dep_ingredient = sacred.Ingredient(
    'dqn_setup_vae_dep')


@dqn_setup_vae_dep_ingredient.config
def dqn_setup_vae_dep_config():
    from_file = True
    file = 'trained_models/muse_last_checkpoint.pth.tar'

    from_mongodb = False

    assert (from_file ^
            from_mongodb) == True, "only one can be true. (xor operation)"
    assert (not from_file) or (
            file is not None), "if from_file, then file cannot be None"


#################
# Eval pipeline #
#################

eval_pipeline_ingredient = sacred.Ingredient('eval_pipeline')


@eval_pipeline_ingredient.config
def eval_pipeline_config():
    eval_episodes = 100

    condition_on_joint = False
    condition_on_image = False
    condition_on_sound = True

    vae_from_file = True
    vae_file = "/mnt/DATA/Research/Laboratory/Research/gmc_no_modification/gmc_code/rl/trained_models/hyperhot/design 1/finetune/gmc_hyperhot_model.pth.tar"
    # vae_file = "/mnt/DATA/Research/Laboratory/Research/gmc_no_modification/gmc_code/rl/trained_models/hyperhot/design 1/gmc_hyperhot_model.pth.tar"
    # vae_file = "/mnt/DATA/Research/Laboratory/Research/gmc_no_modification/gmc_code/rl/trained_models/hyperhot/design 2/gmc_hyperhot2_model.pth.tar"
    # vae_file = "/mnt/DATA/Research/Laboratory/Research/gmc_no_modification/gmc_code/rl/trained_models/hyperhot/design 3/gmc_hyperhot3_model.pth.tar"
    vae_from_mongodb = False

    dqn_from_file = True
    # dqn_file = "/mnt/DATA/Research/Laboratory/Research/muse_gmc/muse/evaluation/atari/hyperhot/trained_models/gmc_pretrain/best_dqn_model.pth.tar"
    dqn_file = "/mnt/DATA/Research/Laboratory/Research/muse_gmc/muse/evaluation/atari/hyperhot/trained_models/gmc_pretrain/best/best_dqn_model.pth.tar"
    # dqn_file = 'trained_models/best_dqn_model.pth.tar'
    # dqn_file = '/mnt/DATA/Research/Laboratory/Research/muse_gmc/muse/evaluation/atari/hyperhot/trained_models/rl_full/design 1/best_dqn_model.pth.tar'
    dqn_from_mongodb = False

    assert (vae_from_file ^ vae_from_mongodb
            ) == True, "only one can be true. (xor operation)"
    assert (not vae_from_file) or (
            vae_file is not None), "if from_file, then file cannot be None"

    assert (dqn_from_file ^
            dqn_from_mongodb) == True, "only one can be true. (xor operation)"
    assert (not dqn_from_file) or (
            dqn_file is not None), "if from_file, then file cannot be None"


########
# CUDA #
########


gmc_hyperhot_named_config = sacred.Ingredient('gmc_hyperhot')


@gmc_hyperhot_named_config.named_config
def gmc_hyperhot():
    rep_model = "gmc"
    image_side = 80
    sound_samples = 1047
    n_stack = 2
    n_enemies = 4
    pacifist_mode = False
    time_limit = 15
    sound_receivers = ["LEFT_BOTTOM", "RIGHT_BOTTOM", "LEFT_SHIP", "RIGHT_SHIP"]
    random_seed = 42
    common_dim = 64
    latent_dim = 40
    checkpoint = "/mnt/DATA/Research/Laboratory/Research/gmc_no_modification/gmc_code/rl/trained_models/hyperhot/design 1/gmc_hyperhot_model.pth.tar"
    loss_type = "joints_as_negatives"
    learner = "dqn"


@gmc_hyperhot_named_config.named_config
def gmc_hyperhot2():
    rep_model = "gmc"
    image_side = 80
    sound_samples = 1047
    n_stack = 2
    n_enemies = 4
    pacifist_mode = False
    time_limit = 15
    sound_receivers = ["LEFT_BOTTOM", "RIGHT_BOTTOM", "TOP_SHIP", "BOTTOM_SHIP"]
    random_seed = 42
    common_dim = 64
    latent_dim = 40
    checkpoint = ""
    loss_type = "joints_as_negatives"
    checkpoint = "/mnt/DATA/Research/Laboratory/Research/gmc_no_modification/gmc_code/rl/trained_models/hyperhot/design 2/gmc_hyperhot2_model.pth.tar"
    learner = "dqn"


@gmc_hyperhot_named_config.named_config
def gmc_hyperhot3():
    rep_model = "gmc"
    image_side = 80
    sound_samples = 1047
    n_stack = 2
    n_enemies = 4
    pacifist_mode = False
    time_limit = 15
    sound_receivers = ["RIGHT_SHIP", "RIGHT_SHIP", "RIGHT_SHIP", "RIGHT_SHIP"]
    random_seed = 42
    common_dim = 64
    latent_dim = 40
    checkpoint = "/mnt/DATA/Research/Laboratory/Research/gmc_no_modification/gmc_code/rl/trained_models/hyperhot/design 1/gmc_hyperhot_model.pth.tar"
    loss_type = "joints_as_negatives"
    learner = "dqn"


@gmc_hyperhot_named_config.named_config
def gmc_hyperhot4():
    image_side = 80
    sound_samples = 1047
    n_stack = 2
    n_enemies = 4
    pacifist_mode = False
    time_limit = 15
    sound_receivers = ["LEFT_BOTTOM", "RIGHT_BOTTOM", "LEFT_BOTTOM", "RIGHT_BOTTOM"]
    random_seed = 42
    common_dim = 64
    latent_dim = 40
    checkpoint = ""
    loss_type = "joints_as_negatives"
    learner = "dqn"


gpu_ingredient = sacred.Ingredient('gpu')


@gpu_ingredient.config
def gpu_config():
    cuda = True


gmc_ingredient = sacred.Ingredient('gmc')


@gmc_ingredient.config
def exp_config():
    env = "hyperhot"
    if env == "hyperhot":
        train_config = gmc_hyperhot()
    if env == "hyperhot2":
        train_config = gmc_hyperhot2()
    if env == "hyperhot3":
        train_config = gmc_hyperhot3()
    if env == "hyperhot4":
        train_config = gmc_hyperhot4()
