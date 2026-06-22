#!/bin/bash

#SBATCH --job-name=GAT_DGI_Att_training            
#SBATCH --output=job_output_%j.txt         
#SBATCH --time=72:00:00                  
#SBATCH --partition=luna-gpu-long
#SBATCH --gres=gpu:4g.40gb:1                    
#SBATCH --mem=32GB                         
#SBATCH --cpus-per-task=8                


module load Anaconda3/2024.02-1
module load cuda/11.8
source ~/my-scratch/miniconda3/etc/profile.d/conda.sh
conda activate GNN

# Set PyTorch CUDA memory allocation configuration
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Diagnostic: GPU details
echo "=== GPU info ==="
nvidia-smi
nvidia-smi -L
echo "================"

cd /home/vascul/vsayyalasomayajula/my-rdisk/r-divb/venkat/Proteomics/PlasmaAAA/GNN_models/


python /home/vascul/vsayyalasomayajula/my-rdisk/r-divb/venkat/Proteomics/PlasmaAAA/GNN_models/InfoGraph/GAT_InfoGraph_Att.py
