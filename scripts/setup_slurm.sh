#!/bin/bash
# Installation and download IPA dictionaries

#SBATCH -c 12
#SBATCH --mem=12GB
#SBATCH -G 1
#SBATCH -p gpu-preempt
#SBATCH --time 01:00:00
#SBATCH -o setup_%j.out
#SBATCH --mail-type END

req_path=../requirements.txt
module load miniconda/22.11.1-1

conda create -y --name sinr_icml_stag python==3.9
conda activate sinr_icml_stag 

pip3 install -r "$req_path"

