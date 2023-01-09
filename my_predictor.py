'''
Execution details (Hybrid Jofre cluster)
activate conda environment: 'tf-gpu'
execute by: XLA_FLAGS=--xla_gpu_cuda_data_dir=/usr/lib/cuda python3 <python_script_name>
'''
"""
PINNS: loss comes from a combination of:
    (1) (Unsup. loss) Equation of State of Real Gas, 
    (2) (Unsup. loss) Equation of Cp of Real Gas, 
    (3) Supervised loss RAE.
NN input is a SCALAR in spatial dimension, num_features per single point (x,y,z), shape [,num_features]
Features used: idx 1,2: 'u', 'TKE_normalized'
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import argparse
import os
import sys

import tensorflow as tf
import numpy as np

from tensorflow.keras import models, layers
# from ScipyOP import optimizer as SciOP # L-BFGS-B optimizer

from my_dataset_builder import get_datasets_prediction
from my_losses import *
from my_models import MLP
from my_parser import get_arguments

# Get arguments
args = get_arguments()

# ----- Datasets ----

dataset_pred, args = get_datasets_prediction(args)

# ----- Loss -----

if args.loss == "MSE":
    loss_func = MSE(args)
elif args.loss == "RSE":
    loss_func = RSE(args)
elif args.loss == "Supervised_PINNS":
    loss_func = Supervised_PINNS(args)
else:
    sys.exit(f"ValueError: Incorrect argument --loss = '{args.loss}'")

# ----- Metrics -----

metric_func = []
for m in args.metrics:
    if m == "MSE":
        metric_func.append(MSE(args))
    elif m == "RAE":
        metric_func.append(RAE(args))
    elif m == "RAE_target_0":
        metric_func.append(RAE_target_0(args))
    elif m == "RAE_target_1":
        metric_func.append(RAE_target_1(args))
    elif m == "RAE_target_2":
        metric_func.append(RAE_target_2(args))
    elif m == "RSE":
        metric_func.append(RSE(args))
    elif m == "RE_RealGasEq":
        metric_func.append(RelError_RealGasEquation(args))
    elif m == "RE_CpEq":
        metric_func.append(RelError_CpEquation(args))
    elif m == "Supervised_PINNS":
        metric_func.append(Supervised_PINNS(args))
    else:
        sys.exit(f"ValueError: Incorrect argument --metric = '{m}'")
args.num_metrics = len(metric_func)

# ----- Model -----

# Model architecture
inp = layers.Input(shape = (args.num_features))
hl = inp
for i in range(args.num_hidden_layers):
    hl = layers.Dense(args.num_neurons_per_layer, activation = args.activation_function)(hl)
out = layers.Dense(args.num_targets, activation = args.activation_function)(hl) # you may try putting activation = act_fun, or None

model = models.Model(inp, out)
print(model.summary())

# ------------------------------- LOAD TRAINED WEIGHTS --------------------------
model.save_weights(args.ckpt_filename_prediction)


# Model + Optimizer + Loss + Metrics
mlp = MLP(model, loss_func=loss_func, metric_func=metric_func, epochs=args.num_epochs, args=args) 

# ----- Training + Validation -----
mlp.predict(dataset_pred, args)
