'''
Execution details (only Núria's cluster)
activate conda environment: 'tf-gpu'
execute by: XLA_FLAGS=--xla_gpu_cuda_data_dir=/usr/lib/cuda python3 <python_script_name>
'''

import argparse

import numpy as np
import seaborn as sns
import tensorflow as tf

from matplotlib import pyplot as plt
from tensorflow.keras import models, layers, optimizers, activations
# from ScipyOP import optimizer as SciOP # L-BFGS-B optimizer


parser = argparse.ArgumentParser(description="PINN_RANS_channel_flow")
parser.add_argument("--ndim", default=3, type=int, help="problem dimensions")
parser.add_argument("--features_idx", default=[0,1,2,3,4], type=str, help="Selected features index")
parser.add_argument("--targets_name", default=['rho',], type=str, help="Selected targets name")
parser.add_argument("--training_filenames", default=['/home/jofre/Students/Nuria_Masclans/datasets/post_processed/59300000_5features_4targets/3d_high_pressure_turbulent_channel_flow_59300000.npz',], type=str, help="List of training filenames (abspath)")
parser.add_argument("--validation_filenames", default=['/home/jofre/Students/Nuria_Masclans/datasets/post_processed/59300000_5features_4targets/3d_high_pressure_turbulent_channel_flow_59300000.npz',], type=str, help="List of validation filenames (abspath)")
parser.add_argument("--spatial_dimension", default=[128,128,128], type=list, help="Spatial discretization, grid of statistics data. Equals the shape of the stored quantities in 'statistic")
parser.add_argument("--num_hidden_layers", default=20, type=int, help="Number of hidden layers of the model")
parser.add_argument("--num_neurons_per_layer", default=20, type=int, help="Number of neurons per layer of the model")
parser.add_argument("--num_epochs", default=500, type=int, help="Number of training epochs")
parser.add_argument("--visualization_step", default=10, type=int, help="") # TODO

args = parser.parse_args()
args.num_features = len(args.features_idx)
args.num_targets  = len(args.targets_name)

# ____________________________________________________________________________
#
#           Import statistics data from args.statistics_filename
# ____________________________________________________________________________


features_tr  = np.zeros(shape = args.spatial_dimension + args.num_features)
features_val = np.zeros(shape = args.spatial_dimension + args.num_features)
targets_tr   = np.zeros(shape = args.spatial_dimension + args.num_targets)
targets_val  = np.zeros(shape = args.spatial_dimension + args.num_targets)
assert len(args.training_filenames) == 1,   'code implemented only for 1 training file' 
assert len(args.validation_filenames) == 1, 'code implemented only for 1 validation file' 
with np.load(args.training_filenames[0]) as f:
    features_data  = f['x']
    args.features_name = f['features_names']
    for ii in range(args.num_features):
        features_tr[:,:,:,ii] = features_data[:,:,:,args.features_idx[ii]]
    for tt in range(args.num_targets):
        targets_tr[:,:,:,tt]  = f[args.targets_name[tt]]
with np.load(args.validation_filenames[0]) as f:
    features_data  = f['x']
    # features_names = f['features_names']
    for ii in range(args.num_features):
        features_val[:,:,:,ii] = features_data[:,:,:,args.features_idx[ii]]
    for tt in range(args.num_targets):
        targets_val[:,:,:,tt]  = f[args.targets_name[tt]]

# Reshape, to get one discretized node as NN input:
features_tr  = features_tr.reshape(-1,  args.num_features)
features_val = features_val.reshape(-1, args.num_features)
targets_tr   = targets_tr.reshape(-1,   args.num_targets)
targets_val  = targets_val.reshape(-1,  args.num_targets)

# Model: Multi-Layer Perceptron
class MLP(models.Model):
    def __init__(self, model, optimizer, sopt=None, epochs=100, **kwargs):
        super(MLP, self).__init__(**kwargs)
        self.model = model # neural network 
        self.optimizer = optimizer # Adam optimizer
        self.sopt = sopt # L-BFGS-B optimizer 
        self.epochs = epochs # number of epochs for training using Adam
        self.hist = []
        self.epoch = 0

    @tf.function
    def train_step(self, x, y_gt):
        with tf.GradientTape() as tape:
            y_pred = self.model(x) # targets predicted!
            loss   = tf.reduce_mean(tf.square(y_gt - y_pred), axis = 0)   # loss

        trainable_vars = self.trainable_variables
        grads = tape.gradient(loss, trainable_vars)

        tf.print('\nloss:', loss, summarize=-1)
        return loss, grads
    
    def fit(self, x, y_gt):
        x    = tf.convert_to_tensor(x, tf.float32)
        y_gt = tf.convert_to_tensor(y_gt, tf.float32)
        
        # --> training using Adam optimizer 
        for epoch in range(self.epochs):
            tf.print('epoch:', self.epoch)
            loss, grads = self.train_step(x, y_gt)
            self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
            self.epoch += 1
            self.hist.append(loss)
            
        # --> training using L-BFGS-B optimizer
        # objective function for Scipy (L-BFGS-B) optimizer
        """
        def func(params_1d):
            self.sopt.assign_params(params_1d)
            tf.print('epoch:', self.epoch)
            loss, grads = self.train_step(x, y_gt)
            grads = tf.dynamic_stitch(self.sopt.idx, grads)
            self.epoch += 1
            self.hist.append(loss)
            return loss.numpy().astype(np.float64), grads.numpy().astype(np.float64)
        self.sopt.minimize(func)
        """
            
        return self.hist
    
    def predict(self, x):
        x = tf.convert_to_tensor(x, tf.float32)
        y_pred = self.model(x)
        return y_pred.numpy()
    

act = activations.tanh
inp = layers.Input(shape = (args.num_features,))
hl = inp
for i in range(args.num_hidden_layers):
    hl = layers.Dense(args.num_neurons_per_layer, activation = act)(hl)
out = layers.Dense(args.num_targets)(hl)

model = models.Model(inp, out)
print(model.summary())

lr = 1e-3
opt = optimizers.Adam(lr)
# sopt = SciOP(model)

# TRAINING
mlp = MLP(model, opt, args.num_epochs) 
hist = mlp.fit(features_tr, targets_tr)

# VALIDATION
targets_val_pred = mlp.predict(features_val)
targets_val_pred_reshape = targets_val_pred.reshape(args.spatial_dimension+[args.num_targets])
print(f'prediction shape (after reshape): {targets_val_pred_reshape.shape}')


# We plot the quantities at the center of the computational domain

palette = sns.color_palette("muted")
target_idx = 0

for feat_idx in range(args.num_features):
    fig = plt.figure()
    sns.kdeplot(data=targets_val[::args.visualization_step, target_idx],color=palette[0],label="ground-truth")
    sns.kdeplot(data=targets_val_pred[::args.visualization_step, feat_idx],color=palette[1],label="prediction")
    plt.xlabel(args.features_name[feat_idx])
    plt.ylabel("kde, density of probability")
    plt.legend()
    plt.savefig("kde_{}".format(args.features_name[feat_idx]))
    plt.close()