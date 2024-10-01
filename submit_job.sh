#!/bin/bash
#SBATCH --job-name=toxicity_analysis 
#SBATCH --output=data/output/log-%j.log              
#SBATCH --error=data/output/error-%j.log             
#SBATCH --gres=gpu:ampere:1              
#SBATCH --mem=64G                        
#SBATCH --cpus-per-task=4                
#SBATCH --time=2-00:00:00                 
#SBATCH --container-image=registry.webis.de/code-teaching/theses/thesis-klueber/toxicity:1.0.0
#SBATCH --container-mounts=/etc/slurm:/etc/slurm,/usr/lib/x86_64-linux-gnu/slurm:/usr/lib/x86_64-linux-gnu/slurm,/run/munge:/run/munge,/mnt/ceph/storage/data-tmp/current/po87xox/thesis-klueber:/tmp

srun --container-workdir=/tmp --container-image=registry.webis.de/code-teaching/theses/thesis-klueber/toxicity:1.0.0 python3 src/toxic-bert/main.py
