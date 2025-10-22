import sacred

########################
#      Scenario        #
########################

scenario_ingredient = sacred.Ingredient("scenario")


@scenario_ingredient.named_config
def pendulum():
    scenario = 'pendulum'
    data_dir = './dataset/'
    image_side = 60
    n_stack = 2
    sound_frequency = 440.
    sound_velocity = 20.
    sound_receivers = ['LEFT_BOTTOM', 'RIGHT_BOTTOM', 'MIDDLE_TOP']
    # sound_receivers = ["LEFT_TOP", "RIGHT_TOP", "MIDDLE_BOTTOM"]

    train_samples = 20000
    test_samples = 2000
    random_seed = 42
    finetune = "sound"
    train_vae = 0


@scenario_ingredient.named_config
def pendulum2():
    scenario = 'pendulum2'
    data_dir = './dataset/'
    image_side = 60
    n_stack = 2
    sound_frequency = 440.
    sound_velocity = 20.
    # sound_receivers = ['LEFT_BOTTOM', 'RIGHT_BOTTOM', 'MIDDLE_TOP']
    sound_receivers = ["LEFT_TOP", "RIGHT_TOP", "MIDDLE_BOTTOM"]

    train_samples = 20000
    test_samples = 2000
    random_seed = 42
    finetune = "sound"
    train_vae = 0

@scenario_ingredient.named_config
def pendulum3():
    scenario = 'pendulum3'
    data_dir = './dataset/'
    image_side = 60
    n_stack = 2
    sound_frequency = 440.
    sound_velocity = 20.
    # sound_receivers = ['LEFT_BOTTOM', 'RIGHT_BOTTOM', 'MIDDLE_TOP']
    sound_receivers = ["LEFT_TOP", "RIGHT_TOP", "MIDDLE_BOTTOM"]

    train_samples = 20000
    test_samples = 2000
    random_seed = 42
    finetune = "sound"
    train_vae = 0

@scenario_ingredient.named_config
def hyperhot():
    scenario = "hyperhot"
    data_dir='./dataset/hyperhot'
    train_samples = 32000
    test_samples = 8000
    image_side = 80
    sound_samples = 1047
    n_stack = 2
    n_enemies = 4
    pacifist_mode = False
    time_limit = 15
    sound_receivers = ["LEFT_BOTTOM", "RIGHT_BOTTOM", "LEFT_SHIP", "RIGHT_SHIP"]
    random_seed = 42
    finetune = "sound"

@scenario_ingredient.named_config
def hyperhot2():
    scenario = "hyperhot2"
    data_dir='./dataset/hyperhot2'
    train_samples = 32000
    test_samples = 8000
    image_side = 80
    sound_samples = 1047
    n_stack = 2
    n_enemies = 4
    pacifist_mode = False
    time_limit = 15
    sound_receivers = ["LEFT_BOTTOM", "RIGHT_BOTTOM", "TOP_SHIP", "BOTTOM_SHIP"]
    random_seed = 42
    finetune = "sound"

@scenario_ingredient.named_config
def hyperhot3():
    scenario = "hyperhot3"
    data_dir='./dataset/hyperhot3'
    train_samples = 32000
    test_samples = 8000
    image_side = 80
    sound_samples = 1047
    n_stack = 2
    n_enemies = 4
    pacifist_mode = False
    time_limit = 15
    sound_receivers = ["LEFT_BOTTOM", "LEFT_BOTTOM", "LEFT_BOTTOM", "LEFT_BOTTOM"]
    random_seed = 42
    finetune = "sound"

@scenario_ingredient.named_config
def hyperhot4():
    scenario = "hyperhot4"
    data_dir='./dataset/hyperhot4'
    train_samples = 32000
    test_samples = 8000
    image_side = 80
    sound_samples = 1047
    n_stack = 2
    n_enemies = 4
    pacifist_mode = False
    time_limit = 15
    sound_receivers = ["LEFT_BOTTOM", "RIGHT_BOTTOM", "TOP_SHIP", "BOTTOM_SHIP"]
    random_seed = 42
    finetune = "sound"

@scenario_ingredient.named_config
def vizdoom0():
    scenario = "vizdoom0"
    data_dir = './dataset/vizdoom0'
    train_samples = 32000
    test_samples = 8000
    random_seed = 42
    finetune = "sound"
    n_stack = 1
    use_sonic_aim_support = False
    use_auto_aim_support = False

@scenario_ingredient.named_config
def vizdoom1():
    scenario = "vizdoom1"
    data_dir = './dataset/vizdoom1'
    train_samples = 32000
    test_samples = 8000
    random_seed = 42
    finetune = "sound"
    n_stack = 1
    use_sonic_aim_support = True
    use_auto_aim_support = False

@scenario_ingredient.named_config
def vizdoom2():
    scenario = "vizdoom2"
    data_dir = './dataset/vizdoom2'
    train_samples = 32000
    test_samples = 8000
    random_seed = 42
    finetune = "sound"
    n_stack = 1

@scenario_ingredient.named_config
def vizdoom3():
    scenario = "vizdoom3"
    data_dir = './dataset/vizdoom3'
    train_samples = 32000
    test_samples = 8000
    random_seed = 42
    finetune = "sound"
    n_stack = 1

@scenario_ingredient.named_config
def vizdoom4():
    scenario = "vizdoom4"
    data_dir = './dataset/vizdoom4'
    train_samples = 32000
    test_samples = 8000
    random_seed = 42
    finetune = "sound"
    n_stack = 1

@scenario_ingredient.named_config
def vizdoom5():
    scenario = "vizdoom5"
    data_dir = './dataset/vizdoom5'
    train_samples = 32000
    test_samples = 8000
    random_seed = 42
    finetune = "sound"
    n_stack = 1