#!/bin/bash
# Installation and download IPA dictionaries

#SBATCH -c 12
#SBATCH --mem=32GB
#SBATCH -G 1
#SBATCH -p gpu-preempt
#SBATCH --time 02:00:00
#SBATCH -o trainneval_%j.out
#SBATCH --mail-type ALL

module load miniconda/22.11.1-1
module load cuda/11.3.1
conda activate sinr_icml

python train_and_evaluate_models.py
