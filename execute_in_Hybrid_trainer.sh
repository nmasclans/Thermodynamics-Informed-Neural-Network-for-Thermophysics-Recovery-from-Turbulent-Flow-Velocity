#!/bin/bash

#SBATCH --output=PINNS_training_%j.out

for HIDDEN_LAYERS in 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20; do
    for NEURONS_PER_HIDDEN_LAYER in 8 16 32 64 128 256; do
        for LEARNING_RATE in 0.001 0.0001; do
            echo ""
            echo "----------------------------------------------------------------------------------"
            echo "----------------------------------------------------------------------------------"
            echo "Hidden Layers: $HIDDEN_LAYERS, Neurons per hidden layer: $NEURONS_PER_HIDDEN_LAYER, Learning rate: $LEARNING_RATE"
            echo "----------------------------------------------------------------------------------"
            echo "----------------------------------------------------------------------------------"
            XLA_FLAGS=--xla_gpu_cuda_data_dir=/usr/lib/cuda python3 my_trainer.py \
		        --num_hidden_layers $HIDDEN_LAYERS \
		        --num_neurons_per_layer $NEURONS_PER_HIDDEN_LAYER \
                --learning_rate $LEARNING_RATE \
	            --num_epochs 30
        done
    done
done
