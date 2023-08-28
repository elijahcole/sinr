import numpy as np
import pandas as pd
import random
import torch
import time
import os
import copy
import json
import tifffile
import h3
import setup

from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import MinMaxScaler

import utils
import models
import datasets

class EvaluatorSNT:
    def __init__(self, train_params, eval_params):
        self.train_params = train_params
        self.eval_params = eval_params
        with open('paths.json', 'r') as f:
            paths = json.load(f)
        D = np.load(os.path.join(paths['snt'], 'snt_res_5.npy'), allow_pickle=True)
        D = D.item()
        self.loc_indices_per_species = D['loc_indices_per_species']
        self.labels_per_species = D['labels_per_species']
        self.taxa = D['taxa']
        self.obs_locs = D['obs_locs']
        self.obs_locs_idx = D['obs_locs_idx']

    def get_labels(self, species):
        species = str(species)
        lat = []
        lon = []
        gt = []
        for hx in self.data:
            cur_lat, cur_lon = h3.h3_to_geo(hx)
            if species in self.data[hx]:
                cur_label = int(len(self.data[hx][species]) > 0)
                gt.append(cur_label)
                lat.append(cur_lat)
                lon.append(cur_lon)
        lat = np.array(lat).astype(np.float32)
        lon = np.array(lon).astype(np.float32)
        obs_locs = np.vstack((lon, lat)).T
        gt = np.array(gt).astype(np.float32)
        return obs_locs, gt

    def run_evaluation(self, model, enc):
        results = {}

        # set seeds:
        np.random.seed(self.eval_params['seed'])
        random.seed(self.eval_params['seed'])

        # evaluate the geo model for each taxon
        results['per_species_average_precision_all'] = np.zeros((len(self.taxa)), dtype=np.float32)

        # get eval locations and apply input encoding
        obs_locs = torch.from_numpy(self.obs_locs).to(self.eval_params['device'])
        loc_feat = enc.encode(obs_locs)

        # get classes to eval
        classes_of_interest = torch.zeros(len(self.taxa), dtype=torch.int64)
        for tt_id, tt in enumerate(self.taxa):
            class_of_interest = np.where(np.array(self.train_params['class_to_taxa']) == tt)[0]
            if len(class_of_interest) != 0:
                classes_of_interest[tt_id] = torch.from_numpy(class_of_interest)

        # generate model predictions for classes of interest at eval locations
        with torch.no_grad():
            loc_emb = model(loc_feat, return_feats=True)
            wt = model.class_emb.weight[classes_of_interest, :]
            pred_mtx = torch.matmul(loc_emb, torch.transpose(wt, 0, 1)).cpu().numpy()

        split_rng = np.random.default_rng(self.eval_params['split_seed'])
        for tt_id, tt in enumerate(self.taxa):

            class_of_interest = np.where(np.array(self.train_params['class_to_taxa']) == tt)[0]
            if len(class_of_interest) == 0:
                # taxa of interest is not in the model
                results['per_species_average_precision_all'][tt_id] = np.nan
            else:
                # generate ground truth labels for current taxa
                cur_loc_indices = np.array(self.loc_indices_per_species[tt_id])
                cur_labels = np.array(self.labels_per_species[tt_id])

                # apply per-species split:
                assert self.eval_params['split'] in ['all', 'val', 'test']
                if self.eval_params['split'] != 'all':
                    num_val = np.floor(len(cur_labels) * self.eval_params['val_frac']).astype(int)
                    idx_rand = split_rng.permutation(len(cur_labels))
                    if self.eval_params['split'] == 'val':
                        idx_sel = idx_rand[:num_val]
                    elif self.eval_params['split'] == 'test':
                        idx_sel = idx_rand[num_val:]
                    cur_loc_indices = cur_loc_indices[idx_sel]
                    cur_labels = cur_labels[idx_sel]

                # extract model predictions for current taxa from prediction matrix
                pred = pred_mtx[cur_loc_indices, tt_id]

                # compute the AP for each taxa
                results['per_species_average_precision_all'][tt_id] = utils.average_precision_score_faster((cur_labels > 0).astype(np.int32), pred)

        valid_taxa = ~np.isnan(results['per_species_average_precision_all'])

        # store results
        per_species_average_precision_valid = results['per_species_average_precision_all'][valid_taxa]
        results['mean_average_precision'] = per_species_average_precision_valid.mean()
        results['num_eval_species_w_valid_ap'] = valid_taxa.sum()
        results['num_eval_species_total'] = len(self.taxa)

        return results

    def report(self, results):
        for field in ['mean_average_precision', 'num_eval_species_w_valid_ap', 'num_eval_species_total']:
            print(f'{field}: {results[field]}')

class EvaluatorIUCN:

    def __init__(self, train_params, eval_params):
        self.train_params = train_params
        self.eval_params = eval_params
        with open('paths.json', 'r') as f:
            paths = json.load(f)
        with open(os.path.join(paths['iucn'], 'iucn_res_5.json'), 'r') as f:
            self.data = json.load(f)
        self.obs_locs = np.array(self.data['locs'], dtype=np.float32)
        self.taxa = [int(tt) for tt in self.data['taxa_presence'].keys()]

    def run_evaluation(self, model, enc):
        results = {}

        results['per_species_average_precision_all'] = np.zeros(len(self.taxa), dtype=np.float32)
        # get eval locations and apply input encoding
        obs_locs = torch.from_numpy(self.obs_locs).to(self.eval_params['device'])
        loc_feat = enc.encode(obs_locs)

        # get classes to eval
        classes_of_interest = torch.zeros(len(self.taxa), dtype=torch.int64)
        for tt_id, tt in enumerate(self.taxa):
            class_of_interest = np.where(np.array(self.train_params['class_to_taxa']) == tt)[0]
            if len(class_of_interest) != 0:
                classes_of_interest[tt_id] = torch.from_numpy(class_of_interest)

        with torch.no_grad():
            # generate model predictions for classes of interest at eval locations
            loc_emb = model(loc_feat, return_feats=True)
            wt = model.class_emb.weight[classes_of_interest, :]
            pred_mtx = torch.matmul(loc_emb, torch.transpose(wt, 0, 1)).cpu().numpy()

        for tt_id, tt in enumerate(self.taxa):
            class_of_interest = np.where(np.array(self.train_params['class_to_taxa']) == tt)[0]

            if len(class_of_interest) == 0:
                # taxa of interest is not in the model
                results['per_species_average_precision_all'][tt_id] = np.nan
            else:
                # extract model predictions for current taxa from prediction matrix
                pred = pred_mtx[:, tt_id]
                gt = np.zeros(obs_locs.shape[0], dtype=np.float32)
                gt[self.data['taxa_presence'][str(tt)]] = 1.0
                # average precision score:
                results['per_species_average_precision_all'][tt_id] = utils.average_precision_score_faster(gt, pred)

        valid_taxa = ~np.isnan(results['per_species_average_precision_all'])

        # store results
        per_species_average_precision_valid = results['per_species_average_precision_all'][valid_taxa]
        results['mean_average_precision'] = per_species_average_precision_valid.mean()
        results['num_eval_species_w_valid_ap'] = valid_taxa.sum()
        results['num_eval_species_total'] = len(self.taxa)
        return results

    def report(self, results):
        for field in ['mean_average_precision', 'num_eval_species_w_valid_ap', 'num_eval_species_total']:
            print(f'{field}: {results[field]}')

class EvaluatorGeoPrior:

    def __init__(self, train_params, eval_params):
        # store parameters:
        self.train_params = train_params
        self.eval_params = eval_params
        with open('paths.json', 'r') as f:
            paths = json.load(f)
        # load vision model predictions:
        self.data = np.load(os.path.join(paths['geo_prior'], 'geo_prior_model_preds.npz'))
        print(self.data['probs'].shape[0], 'total test observations')
        # load locations:
        meta = pd.read_csv(os.path.join(paths['geo_prior'], 'geo_prior_model_meta.csv'))
        self.obs_locs  = np.vstack((meta['longitude'].values, meta['latitude'].values)).T.astype(np.float32)
        # taxonomic mapping:
        self.taxon_map = self.find_mapping_between_models(self.data['model_to_taxa'], self.train_params['class_to_taxa'])
        print(self.taxon_map.shape[0], 'out of', len(self.data['model_to_taxa']), 'taxa in both vision and geo models')

    def find_mapping_between_models(self, vision_taxa, geo_taxa):
        # this will output an array of size N_overlap X 2
        # the first column will be the indices of the vision model, and the second is their
        # corresponding index in the geo model
        taxon_map = np.ones((vision_taxa.shape[0], 2), dtype=np.int32)*-1
        taxon_map[:, 0] = np.arange(vision_taxa.shape[0])
        geo_taxa_arr = np.array(geo_taxa)
        for tt_id, tt in enumerate(vision_taxa):
            ind = np.where(geo_taxa_arr==tt)[0]
            if len(ind) > 0:
                taxon_map[tt_id, 1] = ind[0]
        inds = np.where(taxon_map[:, 1]>-1)[0]
        taxon_map = taxon_map[inds, :]
        return taxon_map

    def convert_to_inat_vision_order(self, geo_pred_ip, vision_top_k_prob, vision_top_k_inds, vision_taxa, taxon_map):
        # this is slow as we turn the sparse input back into the same size as the dense one
        vision_pred = np.zeros((geo_pred_ip.shape[0], len(vision_taxa)), dtype=np.float32)
        geo_pred = np.ones((geo_pred_ip.shape[0], len(vision_taxa)), dtype=np.float32)
        vision_pred[np.arange(vision_pred.shape[0])[..., np.newaxis], vision_top_k_inds] = vision_top_k_prob

        geo_pred[:, taxon_map[:, 0]] = geo_pred_ip[:, taxon_map[:, 1]]

        return geo_pred, vision_pred

    def run_evaluation(self, model, enc):
        results = {}

        # loop over in batches
        batch_start = np.hstack((np.arange(0, self.data['probs'].shape[0], self.eval_params['batch_size']), self.data['probs'].shape[0]))
        correct_pred = np.zeros(self.data['probs'].shape[0])

        for bb_id, bb in enumerate(range(len(batch_start)-1)):
            batch_inds = np.arange(batch_start[bb], batch_start[bb+1])

            vision_probs = self.data['probs'][batch_inds, :]
            vision_inds = self.data['inds'][batch_inds, :]
            gt = self.data['labels'][batch_inds]

            obs_locs_batch = torch.from_numpy(self.obs_locs[batch_inds, :]).to(self.eval_params['device'])
            loc_feat = enc.encode(obs_locs_batch)

            with torch.no_grad():
                geo_pred = model(loc_feat).cpu().numpy()

            geo_pred, vision_pred = self.convert_to_inat_vision_order(geo_pred, vision_probs, vision_inds,
                                                                 self.data['model_to_taxa'], self.taxon_map)

            comb_pred = np.argmax(vision_pred*geo_pred, 1)
            comb_pred = (comb_pred==gt)
            correct_pred[batch_inds] = comb_pred

        results['vision_only_top_1'] = float((self.data['inds'][:, -1] == self.data['labels']).mean())
        results['vision_geo_top_1'] = float(correct_pred.mean())
        return results

    def report(self, results):
        print('Overall accuracy vision only model', round(results['vision_only_top_1'], 3))
        print('Overall accuracy of geo model     ', round(results['vision_geo_top_1'], 3))
        print('Gain                              ', round(results['vision_geo_top_1'] - results['vision_only_top_1'], 3))

class EvaluatorGeoFeature:

    def __init__(self, train_params, eval_params):
        self.train_params = train_params
        self.eval_params = eval_params
        with open('paths.json', 'r') as f:
            paths = json.load(f)
        self.data_path = paths['geo_feature']
        self.country_mask = tifffile.imread(os.path.join(paths['masks'], 'USA_MASK.tif')) == 1
        self.raster_names = ['ABOVE_GROUND_CARBON', 'ELEVATION', 'LEAF_AREA_INDEX', 'NON_TREE_VEGITATED', 'NOT_VEGITATED', 'POPULATION_DENSITY', 'SNOW_COVER', 'SOIL_MOISTURE', 'TREE_COVER']
        self.raster_names_log_transform = ['POPULATION_DENSITY']

    def load_raster(self, raster_name, log_transform=False):
        raster = tifffile.imread(os.path.join(self.data_path, raster_name + '.tif')).astype(np.float32)
        valid_mask = ~np.isnan(raster).copy() & self.country_mask
        # log scaling:
        if log_transform:
            raster[valid_mask] = np.log1p(raster[valid_mask] - raster[valid_mask].min())
        # 0/1 scaling:
        raster[valid_mask] -= raster[valid_mask].min()
        raster[valid_mask] /= raster[valid_mask].max()

        return raster, valid_mask

    def get_split_labels(self, raster, split_ids, split_of_interest):
        # get the GT labels for a subset
        inds_y, inds_x = np.where(split_ids==split_of_interest)
        return raster[inds_y, inds_x]

    def get_split_feats(self, model, enc, split_ids, split_of_interest):
        locs = utils.coord_grid(self.country_mask.shape, split_ids=split_ids, split_of_interest=split_of_interest)
        locs = torch.from_numpy(locs).to(self.eval_params['device'])
        locs_enc = enc.encode(locs)
        with torch.no_grad():
            feats = model(locs_enc, return_feats=True).cpu().numpy()
        return feats

    def run_evaluation(self, model, enc):
        results = {}
        for raster_name in self.raster_names:
            do_log_transform = raster_name in self.raster_names_log_transform
            raster, valid_mask = self.load_raster(raster_name, do_log_transform)
            split_ids = utils.create_spatial_split(raster, valid_mask, cell_size=self.eval_params['cell_size'])
            feats_train = self.get_split_feats(model, enc, split_ids=split_ids, split_of_interest=1)
            feats_test = self.get_split_feats(model, enc, split_ids=split_ids, split_of_interest=2)
            labels_train = self.get_split_labels(raster, split_ids, 1)
            labels_test = self.get_split_labels(raster, split_ids, 2)
            scaler = MinMaxScaler()
            feats_train_scaled = scaler.fit_transform(feats_train)
            feats_test_scaled = scaler.transform(feats_test)
            clf = RidgeCV(alphas=(0.1, 1.0, 10.0), normalize=False, cv=10, fit_intercept=True, scoring='r2').fit(feats_train_scaled, labels_train)
            train_score = clf.score(feats_train_scaled, labels_train)
            test_score = clf.score(feats_test_scaled, labels_test)
            results[f'train_r2_{raster_name}'] = float(train_score)
            results[f'test_r2_{raster_name}'] = float(test_score)
            results[f'alpha_{raster_name}'] = float(clf.alpha_)
        return results

    def report(self, results):
        report_fields = [x for x in results if 'test_r2' in x]
        for field in report_fields:
            print(f'{field}: {results[field]}')
        print(np.mean([results[field] for field in report_fields]))

def launch_eval_run(overrides):

    eval_params = setup.get_default_params_eval(overrides)

    # set up model:
    eval_params['model_path'] = os.path.join(eval_params['exp_base'], eval_params['experiment_name'], eval_params['ckp_name'])
    train_params = torch.load(eval_params['model_path'], map_location='cpu')
    model = models.get_model(train_params['params'])
    model.load_state_dict(train_params['state_dict'], strict=True)
    model = model.to(eval_params['device'])
    model.eval()

    # create input encoder:
    if train_params['params']['input_enc'] in ['env', 'sin_cos_env']:
        raster = datasets.load_env().to(eval_params['device'])
    else:
        raster = None
    enc = utils.CoordEncoder(train_params['params']['input_enc'], raster=raster)

    print('\n' + eval_params['eval_type'])
    t = time.time()
    if eval_params['eval_type'] == 'snt':
        eval_params['split'] = 'test' # val, test, all
        eval_params['val_frac'] = 0.50
        eval_params['split_seed'] = 7499
        evaluator = EvaluatorSNT(train_params['params'], eval_params)
        results = evaluator.run_evaluation(model, enc)
        evaluator.report(results)
    elif eval_params['eval_type'] == 'iucn':
        evaluator = EvaluatorIUCN(train_params['params'], eval_params)
        results = evaluator.run_evaluation(model, enc)
        evaluator.report(results)
    elif eval_params['eval_type'] == 'geo_prior':
        evaluator = EvaluatorGeoPrior(train_params['params'], eval_params)
        results = evaluator.run_evaluation(model, enc)
        evaluator.report(results)
    elif eval_params['eval_type'] == 'geo_feature':
        evaluator = EvaluatorGeoFeature(train_params['params'], eval_params)
        results = evaluator.run_evaluation(model, enc)
        evaluator.report(results)
    else:
        raise NotImplementedError('Eval type not implemented.')
    print(f'evaluation completed in {np.around((time.time()-t)/60, 1)} min')
    return results
