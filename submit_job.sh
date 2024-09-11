#!/bin/bash
#SBATCH --job-name=toxicity_analysis 
#SBATCH --output=data/output/log-%j.log              
#SBATCH --error=data/output/error-%j.log             
#SBATCH --gres=gpu:ampere:1              
#SBATCH --mem=64G                        
#SBATCH --cpus-per-task=4                
#SBATCH --time=2-00:00:00                 
#SBATCH --container-image=registry.webis.de/code-teaching/theses/thesis-klueber/toxicity:0.0.3
#SBATCH --container-mounts=/etc/slurm:/etc/slurm,/usr/lib/x86_64-linux-gnu/slurm:/usr/lib/x86_64-linux-gnu/slurm,/run/munge:/run/munge

# Debug the environment
srun --container-image=registry.webis.de/code-teaching/theses/thesis-klueber/toxicity:0.0.3 \
     python3 -c "import torch; print(torch.cuda.is_available())"

srun python3 src/toxic-bert/main.py

