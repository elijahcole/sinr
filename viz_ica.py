"""
Extracts features from a trained network for each geo location, performs 
dimensionality reduction, and generates an output image.

Different seed values will result in different mappings of locations to colors. 
"""

import torch
import numpy as np
import datasets
import matplotlib.pyplot as plt
import os
from sklearn import decomposition
from skimage import exposure
import json
import utils
import models

# params - specify model of interest here
seed = 2001
with open('paths.json', 'r') as f:
    paths = json.load(f)
num_ds_dims = 3
model_path = 'pretrained_models/model_an_full_input_enc_sin_cos_hard_cap_num_per_class_1000.pt' 
op_file_name = os.path.basename(model_path[:-3]) + '_ica.png'

op_dir = 'visualizations/'
if not os.path.isdir(op_dir):
    os.makedirs(op_dir)

eval_params = {}
if 'device' not in eval_params:
    eval_params['device'] = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
eval_params['model_path'] = model_path

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

# load ocean mask
mask = np.load(os.path.join(paths['masks'], 'ocean_mask.npy'))
mask_inds = np.where(mask.reshape(-1) == 1)[0]

locs = utils.coord_grid(mask.shape)
locs = locs[mask_inds, :]
locs = torch.from_numpy(locs)
locs_enc = enc.encode(locs).to(eval_params['device'])
with torch.no_grad():
    feats = model(locs_enc, return_feats=True).cpu().numpy()

# standardize the features
f_mu = feats.mean(0)
f_std = feats.std(0)
feats = feats - f_mu
feats = feats / f_std
assert not np.any(np.isnan(feats))
assert not np.any(np.isinf(feats))

# downsample features - choose middle time step
print('Performing dimensionality reduction.')
dsf = decomposition.FastICA(n_components=num_ds_dims, random_state=seed, whiten='unit-variance', max_iter=1000)
dsf.fit(feats)

feats_ds = dsf.transform(feats)

# equalize - doing this means there is no need to do the mean normalization
for cc in range(num_ds_dims):
    feats_ds[:, cc] = exposure.equalize_hist(feats_ds[:, cc])

# convert into image
op_im = np.ones((mask.shape[0]*mask.shape[1], num_ds_dims))
op_im[mask_inds] = feats_ds
op_im = op_im.reshape((mask.shape[0], mask.shape[1], num_ds_dims))

# save output
op_path = os.path.join(op_dir, op_file_name)
print('Saving image to: ' + op_path)
plt.imsave(op_path, (op_im*255).astype(np.uint8))
