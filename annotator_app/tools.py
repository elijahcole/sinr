import numpy as np
import matplotlib
matplotlib.use('Agg')
import json
import os

import sys
sys.path
sys.path.append('../')


import alphashape
from shapely.geometry import mapping


import sinr

def load_taxa_metadata(file_path):
    taxa_names_file = open(file_path, "r")
    data = taxa_names_file.read().split("\n")
    data = [dd for dd in data if dd != '']
    taxa_ids = []
    taxa_names = []
    for tt in range(len(data)):
        id, nm = data[tt].split('\t')
        taxa_ids.append(int(id))
        taxa_names.append(nm)
    taxa_names_file.close()
    return dict(zip(taxa_names, taxa_ids))
 

def get_taxa_id_by_name(tax_name: str):
    # TODO get from file
    # TODO check if it exist
    if tax_name.isdigit():
        return int(tax_name)
    taxa_metadata = load_taxa_metadata('taxa_02_08_2023_names.txt')
    return taxa_metadata.get(tax_name)


def save_preds(taxa_id, preds, locs):
    # TODO save to DB
    directory = 'predictions'
    if not os.path.exists(directory):
        os.makedirs(directory)
    np.save(f'{directory}/{taxa_id}_preds', preds)
    np.save(f'{directory}/{taxa_id}_locs', locs)


def get_prediction(eval_params):
    # TODO get preds from DB

    taxa_id = get_taxa_id_by_name(eval_params['taxa_name'])
    eval_params['taxa_id'] = taxa_id
    preds_file = f'predictions/{taxa_id}_preds.npy'
    locs_file = f'predictions/{taxa_id}_locs.npy'

    if os.path.isfile(preds_file) and os.path.isfile(locs_file):
        print(f'Loading saved predictions for Taxa ID #{taxa_id}')
        return np.load(preds_file), np.load(locs_file)

    print(f'Starting generate predictions for Taxa ID #{taxa_id}.\nParams {eval_params}') 
    preds, locs = sinr.generate_prediction(eval_params)
    save_preds(taxa_id, preds, locs)
    return preds, locs


def generate_prediction(eval_params):
    preds, locs = get_prediction(eval_params)

    # switch lats and longs
    locs[:,[1,0]] = locs[:,[0,1]]

    # combine coordinates and predictions
    pred_loc_combined = np.column_stack((locs, preds))
    pred_loc_combined = np.float_(pred_loc_combined)

    # leave only predictions above threshold
    threshold = eval_params['threshold']
    # if a more detailed HeatMap needed, use `pred_loc_combined` for that
    pred_loc_combined = pred_loc_combined[pred_loc_combined[:,2] >= threshold]
    coordinates = pred_loc_combined[:,[0,1]]

    preds_center = coordinates.mean(axis=0)

    hull = alphashape.alphashape(coordinates, 1)
    hull_points = list(mapping(hull)['coordinates'])

    saved_annotation = load_annotation(eval_params)

    return dict(
        preds_center=preds_center.tolist(),
        coordinates=coordinates.tolist(),
        pred_loc_combined=pred_loc_combined.tolist(),
        hull_points=hull_points,
        saved_annotation=saved_annotation['polygons'],
    )


def save_annotation(data):
    taxa_id = get_taxa_id_by_name(data['taxa_name'])
    polygons = data['polygons']
    directory = 'annotations'
    # If the polygons are empty, clear all existing polygons
    if polygons:
        # If the polygons are not empty, add new polygons to the existing ones
        saved_polygons = load_annotation(data)['polygons']
        polygons += saved_polygons
    if not os.path.exists(directory):
        os.makedirs(directory)
    with open(f'{directory}/{taxa_id}.json', 'w') as f:
        json.dump(polygons, f)
    print(f'Saving annotation for taxa ID #{taxa_id}:\nAnnotation:{polygons}')
    return {'polygons': polygons}


def load_annotation(data):
    directory = 'annotations'
    taxa_id = get_taxa_id_by_name(data['taxa_name'])
    polygon_file = f'{directory}/{taxa_id}.json'
    if os.path.isfile(polygon_file):
        with open(polygon_file) as f:
            polygons = json.load(f)
        return {'polygons': polygons}
    return {'polygons': []}
