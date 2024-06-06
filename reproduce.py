import os
import numpy as np
import torch
import shutil

import eval

train_params = {}

"""
save_base
- Name of the directory to save models
"""
train_params["save_base"] = (
    "./reproduce/"  # this will be the root directory of where we save models
)

"""
experiment_name
- Name of the directory where results for this run are saved.
"""
train_params["experiment_name"] = (
    "repr"  # This will be the name of the directory where results for this run are saved.  # this is a subdirectory in save_base
)

# load pretrain: defines pretrain models and paths
# follow changelog.md for instructions
pre_trained_models = [
    {"path": "model_an_full_input_enc_sin_cos_distilled_from_env.pt", "cap": "env"},
    {
        "path": "model_an_full_input_enc_sin_cos_hard_cap_num_per_class_10.pt",
        "cap": 10,
    },
    {
        "path": "model_an_full_input_enc_sin_cos_hard_cap_num_per_class_100.pt",
        "cap": 100,
    },
    {
        "path": "model_an_full_input_enc_sin_cos_hard_cap_num_per_class_1000.pt",
        "cap": 1000,
    },
]

pretrain_path = (
    "/work/pi_vcpartridge_umass_edu/inat/pre_trained_models/" # make sure you have downloaded pretrain models correctly
)

destination_directory = os.path.join(
    train_params["save_base"], train_params["experiment_name"]
)  # copies pretrains into reproduce/repr
os.makedirs(destination_directory, exist_ok=True)
for model in pre_trained_models:
    source_path = os.path.join(pretrain_path, model["path"])
    destination_path = os.path.join(
        destination_directory, os.path.basename(source_path)
    )
    shutil.copy2(source_path, destination_path)

print("Model files successfully transferred.")  # you should see this

# evaluate:
for model in pre_trained_models:
    cap = model["cap"]
    model_name = model[
        "path"
    ]  # reference is confusing,  # FIX this reference, make it model['name'], needs fix from above
    for eval_type in ["snt", "iucn"]: # excludes geo_prior and geo_feature
        eval_params = {}
        eval_params["exp_base"] = train_params[
            "save_base"
        ]  # override eval parameters to use the copied pretrains
        eval_params["experiment_name"] = train_params[
            "experiment_name"
        ]  # eval will generate results into the same folder as the model file, so we had to copy them to keep things clean
        eval_params["eval_type"] = eval_type
        eval_params["ckp_name"] = model_name
        if eval_type == "iucn":
            eval_params["device"] = torch.device("cpu")  # for memory reasons
        cur_results = eval.launch_eval_run(
            eval_params
        )  # will take a few minutes to run
        np.save(  # change script to include cap name to distinguish between models
            os.path.join(
                eval_params["exp_base"],
                train_params["experiment_name"],
                f"results_{eval_type}_{cap}.npy",  # finally makes the results
            ),
            cur_results,
        )

# goto reproduce_tables.ipynb to generate tables from the reproduced results

"""
Note that train_params and eval_params do not contain all of the parameters of interest. Instead,
there are default parameter sets for training and evaluation (which can be found in setup.py).
In this script we create dictionaries of key-value pairs that are used to override the defaults
as needed.
"""
