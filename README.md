# Spatial Implicit Neural Representations for Global-Scale Species Mapping - ICML 2023

Code for training and evaluating global-scale species range estimation models. This code enables the recreation of the results from our ICML 2023 paper [Spatial Implicit Neural Representations for Global-Scale Species Mapping](https://arxiv.org/abs/2306.02564). 

## 🌍 Overview 
Estimating the geographical range of a species from sparse observations is a challenging and important geospatial prediction problem. Given a set of locations where a species has been observed, the goal is to build a model to predict whether the species is present or absent at any location. In this work, we use Spatial Implicit Neural Representations (SINRs) to jointly estimate the geographical range of thousands of species simultaneously. SINRs scale gracefully, making better predictions as we increase the number of training species and the amount of training data per species. We introduce four new range estimation and spatial representation learning benchmarks, and we use them to demonstrate that noisy and biased crowdsourced data can be combined with implicit neural representations to approximate expert-developed range maps for many species.

![Model Prediction](images/sinr_traverse.gif)
<sup>Above we visualize predictions from one of our SINR models trained on data from [iNaturalist](inaturalist.org). On the left we show the learned species embedding space, where each point represents a different species. On the right we see the predicted range of the species corresponding to the red dot on the left.<sup>

## 🔍 Getting Started 

#### Installing Required Packages

1. We recommend using an isolated Python environment to avoid dependency issues. Install the Anaconda Python 3.9 distribution for your operating system from [here](https://www.anaconda.com/download). 

2. Create a new environment and activate it:
```bash
 conda create -y --name sinr_icml python==3.9
 conda activate sinr_icml
```

3. After activating the environment, install the required packages:
```bash
 pip3 install -r requirements.txt
```

#### Data Download and Preparation
Instructions for downloading the data in `data/README.md`.

## 🗺️ Generating Predictions
To generate predictions for a model in the form of an image, run the following command: 
```bash
 python viz_map.py --taxa_id 130714
```
Here, `--taxa_id` is the id number for a species of interest from [iNaturalist](https://www.inaturalist.org/taxa/130714). If you want to generate predictions for a random species, add the `--rand_taxa` instead. 

Note, before you run this command you need to first download the data as described in `data/README.md`. In addition, if you want to evaluate some of the pretrained models from the paper, you need to download those first and place them at `sinr/pretrained_models`. See `web_app/README.md` for more details.

There is also an interactive browser-based demo available in `web_app`.

## 🚅 Training and Evaluating Models

To train and evaluate a model, run the following command:
```bash
 python train_and_evaluate_models.py
```

#### Hyperparameters
Common parameters of interest can be set within `train_and_evaluate_models.py`. All other parameters are exposed in `setup.py`. 

#### Outputs
By default, trained models and evaluation results will be saved to a folder in the `experiments` directory. Evaluation results will also be printed to the command line. 

#### Interactive Model Visualizer
To visualize range predictions from pretrained SINR models, please follow the instructions in `web_app/README.md`. 

##  🙏 Acknowledgements
This project was enabled by data from the Cornell Lab of Ornithology, The International Union for the Conservation of Nature, iNaturalist, NASA, USGS, JAXA, CIESIN, and UC Merced. We are especially indebted to the [iNaturalist](inaturalist.org) and [eBird](https://ebird.org) communities for their data collection efforts. We also thank Matt Stimas-Mackey and Sam Heinrich for their help with data curation. This project was funded by the [Climate Change AI Innovation Grants](https://www.climatechange.ai/blog/2022-04-13-innovation-grants) program, hosted by Climate Change AI with the support of the Quadrature Climate Foundation, Schmidt Futures, and the Canada Hub of Future Earth. This work was also supported by the Caltech Resnick Sustainability Institute and an NSF Graduate Research Fellowship (grant number DGE1745301).  

If you find our work useful in your research please consider citing our paper.  
```
@inproceedings{SINR_icml23,
  title     = {{Spatial Implicit Neural Representations for Global-Scale Species Mapping}},
  author    = {Cole, Elijah and Van Horn, Grant and Lange, Christian and Shepard, Alexander and Leary, Patrick and Perona, Pietro and Loarie, Scott and Mac Aodha, Oisin},
  booktitle = {ICML},
  year = {2023}
}
```

## 📜 Disclaimer
Extreme care should be taken before making any decisions based on the outputs of models presented here. Our goal in this work is to demonstrate the promise of large-scale representation learning for species range estimation, not to provide definitive range maps. Our models are trained on biased data and have not been calibrated or validated beyond the experiments illustrated in the paper. 

