#!/bin/bash

##SBATCH -N 1 
#SBATCH -c 80
#SBATCH --gres=gpu:2
#SBATCH --time=48:00:00
#SBATCH --output=output_MLP_Input2D_%j.out
##SBATCH --error=___%j.err

module purge
module load gcc/8.3.0 cuda/10.2 cudnn/7.6.4 nccl/2.4.8 tensorrt/6.0.1 openmpi/4.0.1 atlas scalapack/2.0.2 fftw/3.3.8 szip/2.1.1 ffmpeg/4.2.1 opencv/4.1.1
module load python/3.7.4_ML
module load tensorflow/2.5.0


for HIDDEN_LAYERS in 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20; do
    for NEURONS_PER_HIDDEN_LAYER in 8 16 32 64 128 256; do
        for LEARNING_RATE in 0.001 0.0001; do
        
        echo ""
        echo "----------------------------------------------------------------------------------"
        echo "----------------------------------------------------------------------------------"
        echo "Hidden Layers: $HIDDEN_LAYERS, Neurons per hidden layer: $NEURONS_PER_HIDDEN_LAYER, Learning rate: $LEARNING_RATE"
        echo "----------------------------------------------------------------------------------"
        echo "----------------------------------------------------------------------------------"
        
        python3 my_trainer.py \
		    --num_hidden_layers $HIDDEN_LAYERS \
		    --num_neurons_per_layer $NEURONS_PER_HIDDEN_LAYER \
            --learning_rate $LEARNING_RATE \
	        --num_epochs 30 \
		    --training_filename "/gpfs/scratch/bsc44/bsc44529/datasets/post_processed/59300000_5features_4targets/3d_high_pressure_turbulent_channel_flow_53900000.npz" \
		    --validation_filename "/gpfs/scratch/bsc44/bsc44529/datasets/post_processed/59300000_5features_4targets/3d_high_pressure_turbulent_channel_flow_59300000.npz"

    done
done
