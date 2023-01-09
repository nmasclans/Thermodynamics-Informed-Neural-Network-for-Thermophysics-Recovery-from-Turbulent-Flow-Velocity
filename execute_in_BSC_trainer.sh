#!/bin/bash

##SBATCH -N 1 
#SBATCH -c 40
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --output=PINNS_training_%j.out
##SBATCH --error=___%j.err

module purge
module load gcc/8.3.0 cuda/10.2 cudnn/7.6.4 nccl/2.4.8 tensorrt/6.0.1 openmpi/4.0.1 atlas scalapack/2.0.2 fftw/3.3.8 szip/2.1.1 ffmpeg/4.2.1 opencv/4.1.1
module load python/3.7.4_ML
module load tensorflow/2.5.0

for SEED in 1 2 3 4 5; do
	echo ""
	echo "----------------------------------------------------------------------------------"
	echo "----------------------------------------------------------------------------------"
	echo "                                   Seed: $SEED"
	echo "----------------------------------------------------------------------------------"
	echo "----------------------------------------------------------------------------------"
	python3 my_trainer.py --seed $SEED --training_filename "/gpfs/scratch/bsc44/bsc44529/datasets/post_processed/59300000_5features_4targets/3d_high_pressure_turbulent_channel_flow_53900000.npz" --validation_filename "/gpfs/scratch/bsc44/bsc44529/datasets/post_processed/59300000_5features_4targets/3d_high_pressure_turbulent_channel_flow_59300000.npz"
done
