#!/bin/bash
#PBS -l walltime=48:00:00
#PBS -l nodes=1:ppn=1
#PBS -r n

module add python/3.5.1
module add CUDA/7.5
module add openblas/0.2.18
source python3.5/bin/activate
cd ~/Hope/Project-IFT6266/DCGAN/
python train_model.py