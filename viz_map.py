"""
Demo that takes an iNaturalist taxa ID as input and generates a prediction 
for each location on the globe and saves the ouput as an image.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import json
import argparse

import utils
import models
import datasets


def main(eval_params):

    # load params
    with open('paths.json', 'r') as f:
        paths = json.load(f)

    # load model
    train_params = torch.load(eval_params['model_path'], map_location='cpu')
    model = models.get_model(train_params['params'])
    model.load_state_dict(train_params['state_dict'], strict=True)
    model = model.to(eval_params['device'])
    model.eval()
    if train_params['params']['input_enc'] in ['env', 'sin_cos_env']:
        raster = datasets.load_env()
    else:
        raster = None
    enc = utils.CoordEncoder(train_params['params']['input_enc'], raster=raster)

    # user specified random taxa
    if eval_params['rand_taxa']: 
        print('Selecting random taxa')
        eval_params['taxa_id'] = np.random.choice(train_params['params']['class_to_taxa'])

    # load taxa of interest 
    if eval_params['taxa_id'] in train_params['params']['class_to_taxa']:
        class_of_interest = train_params['params']['class_to_taxa'].index(eval_params['taxa_id'])
    else:
        print(f'Error: Taxa specified that is not in the model: {eval_params["taxa_id"]}')
        return False
    print(f'Loading taxa: {eval_params["taxa_id"]}')
            
    # load ocean mask
    if eval_params['high_res']:
        mask = np.load(os.path.join(paths['masks'], 'ocean_mask_hr.npy'))
    else:
        mask = np.load(os.path.join(paths['masks'], 'ocean_mask.npy'))
    mask_inds = np.where(mask.reshape(-1) == 1)[0]
        
    # generate input features
    locs = utils.coord_grid(mask.shape)
    if not eval_params['disable_ocean_mask']:
        locs = locs[mask_inds, :]
    locs = torch.from_numpy(locs)
    locs_enc = enc.encode(locs).to(eval_params['device'])
            
    # make prediction
    with torch.no_grad():
        preds = model(locs_enc, return_feats=False, class_of_interest=class_of_interest).cpu().numpy()

    # threshold predictions
    if eval_params['threshold'] > 0:
        print(f'Applying threshold of {eval_params["threshold"]} to the predictions.')
        preds[preds<eval_params['threshold']] = 0.0
        preds[preds>=eval_params['threshold']] = 1.0
        
    # mask data
    if not eval_params['disable_ocean_mask']:
        op_im = np.ones((mask.shape[0] * mask.shape[1])) * np.nan  # set to NaN
        op_im[mask_inds] = preds
    else:
        op_im = preds
        
    # reshape and create masked array for visualization
    op_im = op_im.reshape((mask.shape[0], mask.shape[1]))
    op_im = np.ma.masked_invalid(op_im) 

    # set color for masked values
    cmap = plt.cm.plasma
    cmap.set_bad(color='none')  
    if eval_params['set_max_cmap_to_1']:
        vmax = 1.0
    else:
        vmax = np.max(op_im)
    
    # save image
    save_loc = os.path.join(eval_params['op_path'], str(eval_params['taxa_id']) + '_map.png')
    print(f'Saving image to {save_loc}')
    plt.imsave(fname=save_loc, arr=op_im, vmin=0, vmax=vmax, cmap=cmap)
    
    return True


if __name__ == '__main__':

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        
    info_str = '\nDemo that takes an iNaturalist taxa ID as input and ' + \
               'generates a predicted range for each location on the globe ' + \
               'and saves the ouput as an image.\n\n' + \
               'Warning: these estimated ranges should be validated before use.'  
               
    parser = argparse.ArgumentParser(usage=info_str)
    parser.add_argument('--model_path', type=str, default='./pretrained_models/model_an_full_input_enc_sin_cos_hard_cap_num_per_class_1000.pt')
    parser.add_argument('--taxa_id', type=int, default=130714, help='iNaturalist taxon ID.')
    parser.add_argument('--threshold', type=float, default=-1, help='Threshold the range map [0, 1].')
    parser.add_argument('--op_path', type=str, default='./images/', help='Location where the output image will be saved.')
    parser.add_argument('--rand_taxa', action='store_true', help='Select a random taxa.')
    parser.add_argument('--high_res', action='store_true', help='Generate higher resolution output.')
    parser.add_argument('--disable_ocean_mask', action='store_true', help='Do not use an ocean mask.')
    parser.add_argument('--set_max_cmap_to_1', action='store_true', help='Consistent maximum intensity ouput.')
    parser.add_argument('--device', type=str, default=device, help='cpu or cuda')
    eval_params = vars(parser.parse_args())

    if not os.path.isdir(eval_params['op_path']):
        os.makedirs(eval_params['op_path'])
        
    main(eval_params)
