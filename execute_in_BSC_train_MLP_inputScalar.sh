#!/bin/bash

##SBATCH -N 1 
#SBATCH -c 40
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --output=output_%j.out
##SBATCH --error=results/regression_HPTunning_Experiment_7_BSC_%j.err

module purge
module load gcc/8.3.0 cuda/10.2 cudnn/7.6.4 nccl/2.4.8 tensorrt/6.0.1 openmpi/4.0.1 atlas scalapack/2.0.2 fftw/3.3.8 szip/2.1.1 ffmpeg/4.2.1 opencv/4.1.1
module load python/3.7.4_ML
module load tensorflow/2.5.0


for HIDDEN_LAYERS in 1 2; do
    for NEURONS_PER_HIDDEN_LAYER in 8; do
        
        echo ""
        echo "----------------------------------------------------------------------------------"
        echo "----------------------------------------------------------------------------------"
        echo "        Hidden Layers: $HIDDEN_LAYERS, Neurons per hidden layer: $NEURONS_PER_HIDDEN_LAYER"
        echo "----------------------------------------------------------------------------------"
        echo "----------------------------------------------------------------------------------"
        
        XLA_FLAGS=--xla_gpu_cuda_data_dir=/usr/lib/cuda python3 train_MLP_InputScalar.py --num_hidden_layers $HIDDEN_LAYERS --num_neurons_per_layer $NEURONS_PER_HIDDEN_LAYER --num_epochs 1
    done
done