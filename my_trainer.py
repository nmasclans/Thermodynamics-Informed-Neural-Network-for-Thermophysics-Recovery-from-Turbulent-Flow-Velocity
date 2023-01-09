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

from tensorflow.keras import models, layers, optimizers, activations, initializers, regularizers
# from ScipyOP import optimizer as SciOP # L-BFGS-B optimizer

from my_dataset_builder import get_datasets
from my_losses import *
from my_models import MLP
from my_parser import get_arguments
from my_early_stopping import my_EarlyStopping 
from my_scheduler import my_LearningRateScheduler

# Get arguments
args = get_arguments()

# ----- Datasets ----

dataset_tr, dataset_val, args = get_datasets(args)

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

# ----- Optimizer -----

if args.optimizer == 'Adam':
    opt = optimizers.Adam(args.learning_rate)
else:
    sys.exit(f"argument --optimizer is {args.optimizer}, accepted values: 'Adam'")
# sopt = SciOP(model)

# ----- Model -----

# Initializer: activation function type and distribution
act_fun  = args.activation_function
in_type  = args.initializer_type 
if act_fun == "relu":
    if in_type == "uniform":
        initializer = initializers.HeUniform(seed=args.seed)
    elif in_type == "normal":
        initializer = initializers.HeNormal(seed=args.seed)
    else:
        sys.exit(f"argument --initializer_type is {in_type}, accepted values: 'uniform', 'random'")
elif act_fun == 'tanh':
    if in_type == "uniform":
        initializer = initializers.GlorotUniform(seed=args.seed)
    elif in_type  == "normal":
        initializer = initializers.GlorotNormal(seed=args.seed)
    else:
        sys.exit(f"argument --initializer_type is {in_type}, accepted values: 'uniform', 'random'")
else:
    sys.exit(f"argument --activation_function is {act_fun}, accepted values: 'relu', 'tanh'")

# Regularizer: regularizer type and factor
reg_type = args.regularizer_type
reg_fact = args.regularizer_factor
if reg_type is None:
    regularizer = None
elif reg_type == "L1":
    regularizer = regularizers.L1(l1 = reg_fact)
elif reg_type == "L2":
    regularizer = regularizers.L2(l2 = reg_fact)
else:
    sys.exit(f"argument --regularizer_type is {reg_type}, accepted values: None, 'L1', 'L2'")

# Model architecture
inp = layers.Input(shape = (args.num_features))
hl = inp
for i in range(args.num_hidden_layers):
    hl = layers.Dense(args.num_neurons_per_layer, activation = act_fun, kernel_initializer=initializer, kernel_regularizer=regularizer)(hl)
out = layers.Dense(args.num_targets, activation = act_fun, kernel_initializer = initializer)(hl) # you may try putting activation = act_fun, or None

model = models.Model(inp, out)
print(model.summary())

# Save initial weights (random initialization)
model.save_weights('weights_initialized')

# Early Stopping
early_stopping = my_EarlyStopping(min_delta=args.early_stopping_min_delta, 
    patience=args.early_stopping_patience, start_from_epoch=args.early_stopping_start_from_epoch)

# Learning Rate Scheduler
lr_decay = args.lr_decay
if lr_decay is None or lr_decay == "None":
    lr_scheduler = None
    print(f"No learning rate scheduler is defined. Learning rate will have a constant value of {args.learning_rate}")
elif lr_decay == 'exp':
    lr_0       = args.learning_rate
    decay_rate = args.lr_decay_rate
    decay_step = args.lr_decay_step
    def func_lr_decay(x): 
        return lr_0*np.power(decay_rate,(x/decay_step))
    lr_scheduler = my_LearningRateScheduler(scheduler_function=func_lr_decay, 
        optimizer=opt, call_frequency = args.lr_scheduler_call_frequency, 
        num_batches_per_epoch = args.num_batches_per_epoch, verbose=args.lr_verbose)
    print(f"Learning rate scheduler defined. Learning rate will decay exponentially, with:" \
          f"\n    initial learning rate: {lr_0:.4e}" \
          f"\n    decay rate: {decay_rate:.4e}" \
          f"\n    decay step: {decay_step:.4e}" \
          )
else:
    sys.exit(f"argument --lr_decay is {lr_decay}, accepted values: None, 'None', 'exp'")

# Model + Optimizer + Loss + Metrics
mlp = MLP(model, opt, loss_func=loss_func, metric_func=metric_func, \
    epochs=args.num_epochs, early_stopping=early_stopping, lr_scheduler=lr_scheduler, args=args) 


# ----- Training + Validation -----
mlp.train_and_validate(dataset_tr, dataset_val, args)
