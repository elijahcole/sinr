#!/bin/bash
# Installation and download IPA dictionaries

#SBATCH -c 12
#SBATCH --mem=4GB
#SBATCH -G 1
#SBATCH -p gpu-preempt
#SBATCH --time 01:00:00
#SBATCH -o test_%j.out
#SBATCH --mail-type END

module load miniconda/22.11.1-1
conda activate sinr_icml

echo "Success"
