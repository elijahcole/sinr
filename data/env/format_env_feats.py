import tifffile
import glob
import numpy as np
import os

# Download the 19 bioclimatic variables and elevation at the 5 minute resolution from: https://www.worldclim.org/data/worldclim21.html
# Unzip these files and place them in the "env" folder alongside this python file then run this file.
files = np.sort(glob.glob('*.tif'))

# gather tiff files
assert len(files) > 0
ele_files = [ff for ff in files if 'elev.tif' in ff]
bio_files = np.array([ff for ff in files if '_bio_' in ff])
idx = np.array([int(ff.split('_bio_')[1].split('.')[0]) for ff in bio_files])
bio_files = list(bio_files[np.argsort(idx)])
files = ele_files + bio_files

ims = []
for ff in files: # process into numpy array
    im = tifffile.imread(ff)
    im = im.astype(np.float64)
    fname = os.path.basename(ff)
    # remove outliers
    im[im < -1e6] = np.nan
    mask = ~np.isnan(im)
    # normalize
    im[mask] -= im[mask].mean()
    im[mask] /= im[mask].std()

    ims.append(im)

# want op to be H W C
ims_op = np.zeros((ims[0].shape[0], ims[0].shape[1], len(ims)), dtype=np.float16)
for ii in range(len(ims)):
    ims_op[:,:,ii] = ims[ii].astype(np.float16)
# save bioclimatic data as numpy array
np.save('bioclim_elevation_scaled', ims_op)

