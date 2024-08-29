#!/bin/bash
#SBATCH --job-name=toxicity_analysis 
#SBATCH --output=log-%j.log              
#SBATCH --error=error-%j.log             
#SBATCH --gres=gpu:ampere:1              
#SBATCH --mem=32G                        
#SBATCH --cpus-per-task=4                
#SBATCH --time=2-00:00:00                 
#SBATCH --container-image=pytorch/pytorch:2.1.1-cuda12.1-cudnn8-runtime

srun src/
srun python3 main.py
