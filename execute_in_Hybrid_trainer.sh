#!/bin/bash

#SBATCH --output=PINNS_training_%j.out

for HIDDEN_LAYERS in 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20; do
    for NEURONS_PER_HIDDEN_LAYER in 8 16 32 64 128; do
        for BATCH_SIZE in 16 64; do
            for SEED in 1 2 3 4 5; do
                echo ""
                echo "----------------------------------------------------------------------------------"
                echo "----------------------------------------------------------------------------------"
                echo "Hidden Layers: $HIDDEN_LAYERS, Neurons per hidden layer: $NEURONS_PER_HIDDEN_LAYER, Seed: $SEED"
                echo "----------------------------------------------------------------------------------"
                echo "----------------------------------------------------------------------------------"
                XLA_FLAGS=--xla_gpu_cuda_data_dir=/usr/lib/cuda python3 my_trainer.py \
                    --num_hidden_layers $HIDDEN_LAYERS \
                    --num_neurons_per_layer $NEURONS_PER_HIDDEN_LAYER \
                    --learning_rate 0.001 \
                    --lr_decay "exp" \
                    --seed $SEED \
                    --batch_size $BATCH_SIZE \
                    --num_epochs 30 \
                    --loss "Supervised_PINNS"
        done
    done
done
