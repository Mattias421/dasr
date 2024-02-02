#!/bin/bash

# Parameters
#SBATCH --account=dcs-res
#SBATCH --constraint=''
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --job-name=dasr_darksalmonPidgeotto
#SBATCH --mem=64GB
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --open-mode=append
#SBATCH --output=/mnt/fastdata/acq22mc/exp/dasr/outputs/xps/darksalmonPidgeotto/submitit/%j_0_log.out
#SBATCH --partition=dcs-gpu
#SBATCH --signal=USR2@90
#SBATCH --time=4320
#SBATCH --wckey=submitit

# setup
module load SoX/14.4.2-GCC-8.3.0
module load Anaconda3/5.3.0
source activate speech-diff

# command
export SUBMITIT_EXECUTOR=slurm
srun --unbuffered --output /mnt/fastdata/acq22mc/exp/dasr/outputs/xps/darksalmonPidgeotto/submitit/%j_%t_log.out --export=ALL /fastdata/acq22mc/anaconda/.envs/speech-diff/bin/python -u -m submitit.core._submit /mnt/fastdata/acq22mc/exp/dasr/outputs/xps/darksalmonPidgeotto/submitit
