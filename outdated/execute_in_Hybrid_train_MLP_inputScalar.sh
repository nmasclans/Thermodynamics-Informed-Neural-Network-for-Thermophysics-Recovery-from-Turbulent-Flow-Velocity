#!/bin/bash


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