#!/bin/bash

#SBATCH --job-name=clip_2019
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=30G
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu-vram-48gb
#SBATCH --time=14:00:00

conda activate clip
python /work/mburmest/bachelorarbeit/main.py