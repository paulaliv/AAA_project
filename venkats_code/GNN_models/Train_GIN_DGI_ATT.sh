#!/bin/bash

#SBATCH --job-name=GIN_DGI_Att_training            
#SBATCH --output=job_output_%j.txt         
#SBATCH --time=24:00:00                  
#SBATCH --partition=luna-gpu-long
#SBATCH --gres=gpu:4g.40gb:1                  
#SBATCH --mem=32GB                         
#SBATCH --cpus-per-task=8             


module load Anaconda3/2024.02-1
module load cuda/12.8
source ~/my-scratch/miniconda3/etc/profile.d/conda.sh
conda activate GNN

cd /home/vascul/vsayyalasomayajula/my-rdisk/r-divb/venkat/Proteomics/PlasmaAAA/GNN_models/


python /home/vascul/vsayyalasomayajula/my-rdisk/r-divb/venkat/Proteomics/PlasmaAAA/GNN_models/GIN_DGI_Att.py
