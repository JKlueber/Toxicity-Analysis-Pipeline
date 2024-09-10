#!/bin/bash
#SBATCH --job-name=toxicity_analysis 
#SBATCH --output=log-%j.log              
#SBATCH --error=error-%j.log             
#SBATCH --gres=gpu:ampere:1              
#SBATCH --mem=32G                        
#SBATCH --cpus-per-task=4                
#SBATCH --time=2-00:00:00                 
#SBATCH --container-image=registry.webis.de/code-teaching/theses/thesis-klueber/toxicity:0.0.1

srun src/
srun python3 main.py
