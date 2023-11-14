import gradio as gr
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json
import os
import torch

import sys
sys.path
sys.path.append('../')
import utils
import models
import datasets


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
    return dict(zip(taxa_ids, taxa_names))
 
    
def generate_prediction(taxa_id, selected_model, settings, threshold):
            
    # select the model to use
    if selected_model == 'AN_FULL max 10':
        model_path = '../pretrained_models/model_an_full_input_enc_sin_cos_hard_cap_num_per_class_10.pt'
    elif selected_model == 'AN_FULL max 100':
        model_path = '../pretrained_models/model_an_full_input_enc_sin_cos_hard_cap_num_per_class_100.pt'
    elif selected_model == 'AN_FULL max 1000':
        model_path = '../pretrained_models/model_an_full_input_enc_sin_cos_hard_cap_num_per_class_1000.pt'
    elif selected_model == 'Distilled env model':
        model_path = '../pretrained_models/model_an_full_input_enc_sin_cos_distilled_from_env.pt'

    # load params
    with open('../paths.json', 'r') as f:
        paths = json.load(f)
                
    # configs
    eval_params = {}
    eval_params['device'] = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    eval_params['model_path'] = model_path
    eval_params['taxa_id'] = int(taxa_id)
    eval_params['rand_taxa'] = 'Random taxa' in settings
    eval_params['set_max_cmap_to_1'] = False   
    eval_params['disable_ocean_mask'] = 'Disable ocean mask' in settings
    eval_params['threshold'] = threshold if 'Threshold' in settings else -1.0
        
    # load model
    train_params = torch.load(eval_params['model_path'], map_location='cpu')
    model = models.get_model(train_params['params'])
    model.load_state_dict(train_params['state_dict'], strict=True)
    model = model.to(eval_params['device'])
    model.eval()
    if train_params['params']['input_enc'] in ['env', 'sin_cos_env']:
        raster = datasets.load_env(norm=train_params['params']['env_norm'])
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
        fig = plt.figure()
        plt.imshow(np.zeros((1,1)), vmin=0, vmax=1.0, cmap=plt.cm.plasma)
        plt.axis('off')
        plt.tight_layout()    
        op_html = f'<h2><a href="https://www.inaturalist.org/taxa/{eval_params["taxa_id"]}" target="_blank">{eval_params["taxa_id"]}</a></h2> Error: specified taxa is not in the model.'    
            
        return op_html, fig, eval_params['taxa_id']
    print(f'Loading taxa: {eval_params["taxa_id"]}')

    # load ocean mask
    mask = np.load(os.path.join('../', paths['masks'], 'ocean_mask.npy'))
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
    if eval_params['set_max_cmap_to_1']:
        vmax = 1.0
    else:
        vmax = np.max(op_im)

    # set color for masked values
    cmap = plt.cm.plasma
    cmap.set_bad(color='none')  

    plt.rcParams['figure.figsize'] = 24,12
    fig = plt.figure()
    plt.imshow(op_im, vmin=0, vmax=vmax, cmap=cmap)
    plt.axis('off')
    plt.tight_layout()
    
    # generate html for ouput display
    taxa_name_str = taxa_names[eval_params['taxa_id']]
    op_html = f'<h2><a href="https://www.inaturalist.org/taxa/{eval_params["taxa_id"]}" target="_blank">{taxa_name_str}</a></h2> (click for more info)'    
    return op_html, fig, eval_params['taxa_id']


# load metadata
taxa_names = load_taxa_metadata('taxa_02_08_2023_names.txt')  


with gr.Blocks(title="SINR Demo") as demo:
    top_text = "Visualization code to explore species range predictions "\
               "from Spatial Implicit Neural Representation (SINR) models from "\
               "[our](https://arxiv.org/abs/2306.02564) ICML 2023 paper." 
    gr.Markdown("# SINR Visualization Demo")
    gr.Markdown(top_text)
    
    with gr.Row():
        selected_taxa = gr.Number(label="Taxa ID", value=130714)
        select_model = gr.Dropdown(["AN_FULL max 10", "AN_FULL max 100", "AN_FULL max 1000", "Distilled env model"],
                                    value="AN_FULL max 1000", label="Model")
    with gr.Row():
        settings = gr.CheckboxGroup(["Random taxa", "Disable ocean mask", "Threshold"], label="Settings")
        threshold = gr.Slider(0, 1, 0, label="Threshold")

    with gr.Row():
        submit_button = gr.Button("Run Model")

    with gr.Row():
        output_text = gr.HTML(label="Species Name:")
        
    with gr.Row():
        output_image = gr.Plot(label="Predicted occupancy")

    end_text = "**Note:** Extreme care should be taken before making any decisions "\
               "based on the outputs of models presented here. "\
               "The goal of this work is to demonstrate the promise of large-scale "\
               "representation learning for species range estimation.  "\
               "Our models are trained on biased data and have not been calibrated "\
               "or validated beyondthe experiments illustrated in the paper."
    gr.Markdown(end_text)
                
    submit_button.click(
        fn = generate_prediction,
        inputs=[selected_taxa, select_model, settings, threshold],
        outputs=[output_text, output_image, selected_taxa]
    )

demo.launch()  
