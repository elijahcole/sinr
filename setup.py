import copy
import torch

def apply_overrides(params, overrides):
    params = copy.deepcopy(params)
    for param_name in overrides:
        if param_name not in params:
            print(f'override failed: no parameter named {param_name}')
            raise ValueError
        params[param_name] = overrides[param_name]
    return params

def get_default_params_train(overrides={}):

    params = {}

    '''
    misc
    '''
    params['device'] = 'cuda' # cuda, cpu
    params['save_base'] = './experiments/'
    params['experiment_name'] = 'demo'
    params['timestamp'] = False

    '''
    data
    '''
    params['species_set'] = 'all' # all, snt_birds
    params['hard_cap_seed'] = 9472
    params['hard_cap_num_per_class'] = -1 # -1 for no hard capping
    params['aux_species_seed'] = 8099
    params['num_aux_species'] = 0 # for snt_birds case, how many other species to add in

    '''
    data files
    '''
    params['obs_file'] = 'geo_prior_train.csv'
    params['taxa_file'] = 'geo_prior_train_meta.json'

    '''
    model
    '''
    params['model'] = 'ResidualFCNet'  # ResidualFCNet, LinNet
    params['num_filts'] = 256  # embedding dimension
    params['input_enc'] = 'sin_cos' # sin_cos, env, sin_cos_env
    params['depth'] = 4

    '''
    loss
    '''
    params['loss'] = 'an_full' # an_full, an_ssdl, an_slds
    params['pos_weight'] = 2048

    '''
    optimization
    '''
    params['batch_size'] = 2048
    params['lr'] = 0.0005
    params['lr_decay'] = 0.98
    params['num_epochs'] = 10

    '''
    saving
    '''
    params['log_frequency'] = 512

    params = apply_overrides(params, overrides)

    return params

def get_default_params_eval(overrides={}):

    params = {}

    '''
    misc
    '''
    params['device'] = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    params['seed'] = 2022
    params['exp_base'] = './experiments'
    params['ckp_name'] = 'model.pt'
    params['eval_type'] = 'snt' # snt, iucn, geo_prior, geo_feature
    params['experiment_name'] = 'demo'

    '''
    geo prior
    '''
    params['batch_size'] = 2048

    '''
    geo feature
    '''
    params['cell_size'] = 25

    params = apply_overrides(params, overrides)

    return params
