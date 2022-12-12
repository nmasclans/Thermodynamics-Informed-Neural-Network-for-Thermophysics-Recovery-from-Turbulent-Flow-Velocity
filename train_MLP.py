'''
Execution details (Hybrid Jofre cluster)
activate conda environment: 'tf-gpu'
execute by: XLA_FLAGS=--xla_gpu_cuda_data_dir=/usr/lib/cuda python3 <python_script_name>
'''

import argparse
import os
import sys

import numpy as np
import seaborn as sns
import tensorflow as tf

from matplotlib import pyplot as plt
from tensorflow.keras import models, layers, optimizers, activations
# from ScipyOP import optimizer as SciOP # L-BFGS-B optimizer

os.environ['CUDA_VISIBLE_DEVICES']  = '0'

parser = argparse.ArgumentParser(description="PINN_RANS_channel_flow")
parser.add_argument("--ndim", default=3, type=int, help="problem dimensions")
parser.add_argument("--features_idx", default=[0,1,2,3,4], type=str, help="Selected features index")
parser.add_argument("--targets_name", default=['rho',], type=str, help="Selected targets name")
parser.add_argument("--training_filenames", default=['/home/jofre/Students/Nuria_Masclans/datasets/post_processed/59300000_5features_4targets/3d_high_pressure_turbulent_channel_flow_53900000.npz',], type=str, help="List of training filenames (abspath)")
parser.add_argument("--validation_filenames", default=['/home/jofre/Students/Nuria_Masclans/datasets/post_processed/59300000_5features_4targets/3d_high_pressure_turbulent_channel_flow_59300000.npz',], type=str, help="List of validation filenames (abspath)")
parser.add_argument("--spatial_dimension", default=[128,128,128], type=list, help="Spatial discretization, grid of statistics data. Equals the shape of the stored quantities in 'statistic")
parser.add_argument("--learning_rate", default=1e-3, type=float, help="Learning rate parameter of optimizer")
parser.add_argument("--loss", default="MSE", type=str, help="Loss function name")
parser.add_argument("--num_hidden_layers", default=3, type=int, help="Number of hidden layers of the model")
parser.add_argument("--num_neurons_per_layer", default=8, type=int, help="Number of neurons per layer of the model")
parser.add_argument("--activation_function", default="relu", type=str, help="Activation function (relu, tanh)")
parser.add_argument("--num_epochs", default=20, type=int, help="Number of training epochs")
parser.add_argument("--batch_size", default=16, type=int, help="Batch size (recomended to be multiple of 8)")
parser.add_argument("--visualization_step", default=10, type=int, help="") # TODO
parser.add_argument("--features_limits", 
    default={'y':[0.0,0.0002],'u':[0.0,3.5],'TKE_normalized':[0.0,0.2],'vorticity_magn_normalized':[0.0,55.0],'enstrophy_normalized':[0.0,1400.0]},
    type=dict, help="features minimum and maximum values, used for data normalization"
)
parser.add_argument("--targets_limits", 
    default={'c_p':[1500.0,3900.0],'rho':[145.0,850.0],'T':[90.0,200.0]},
    type=dict, help="targets minimum and maximum values, used for data normalization"
)

args = parser.parse_args()
args.num_features = len(args.features_idx)
args.num_targets  = len(args.targets_name)

# Results Visualizaton
# We plot the quantities at the center of the computational domain
def visualize_prediction(y_gt, y_pred, epoch):
    y_gt_0   = y_gt[:,0]
    y_pred_0 = y_pred[:,0]
    palette = sns.color_palette("muted")
    target_idx = 0

    fig = plt.figure()
    sns.histplot(data=y_gt_0,color=palette[0],bins=100,label="ground-truth")
    sns.histplot(data=y_pred_0,color=palette[1],bins=100,label="prediction")
    plt.ylabel("rho kde, density of probability")
    plt.legend()
    plt.savefig(f"rho_kde_epoch_{epoch}")
    plt.close()

# ____________________________________________________________________________
#
#           Import statistics data from args.statistics_filename
# ____________________________________________________________________________


features_tr  = np.zeros(shape = args.spatial_dimension + [args.num_features,])
features_val = np.zeros(shape = args.spatial_dimension + [args.num_features,])
targets_tr   = np.zeros(shape = args.spatial_dimension + [args.num_targets,])
targets_val  = np.zeros(shape = args.spatial_dimension + [args.num_targets,])
args.features_name = []
assert len(args.training_filenames) == 1,   'code implemented only for 1 training file' 
assert len(args.validation_filenames) == 1, 'code implemented only for 1 validation file' 
with np.load(args.training_filenames[0]) as f:
    features_data  = f['x']
    all_features_name = f['features_names']
    for ii in range(args.num_features):
        features_tr[:,:,:,ii] = features_data[:,:,:,args.features_idx[ii]]
        args.features_name.append(all_features_name[args.features_idx[ii]])
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
args.targets_val_shape = targets_val.shape
print(f"\nShape training features: {features_tr.shape}")
print(f"Shape training targets: {targets_tr.shape}")

# Normalize features and targets data
for feat_idx in range(args.num_features):
    feat_name = args.features_name[feat_idx]
    feat_min  = args.features_limits[feat_name][0]
    feat_max  = args.features_limits[feat_name][1]
    assert (feat_max-feat_min) > 0
    features_tr[:,feat_idx] = (features_tr[:,feat_idx]-feat_min)/(feat_max-feat_min)   - 0.5
    features_val[:,feat_idx] = (features_val[:,feat_idx]-feat_min)/(feat_max-feat_min) - 0.5
    print("\nMin-Max Scaler to feature '{}', from (min,max)=({:.4f},{:.4f}) to (0,1)".format(feat_name, feat_min, feat_max))
    print(f"Min-Max after scaling,  training  dataset: ({features_tr[:,feat_idx].min():.4f},{features_tr[:,feat_idx].max():.4f})")
    print(f"Min-Max after scaling, validation dataset: ({features_val[:,feat_idx].min():.4f},{features_val[:,feat_idx].max():.4f})")
for targ_idx in range(args.num_targets):
    targ_name = args.targets_name[targ_idx]
    targ_min  = args.targets_limits[targ_name][0]
    targ_max  = args.targets_limits[targ_name][1]
    assert (targ_max-targ_min) > 0
    targets_tr[:,targ_idx] = (targets_tr[:,targ_idx]-targ_min)/(targ_max-targ_min)   - 0.5
    targets_val[:,targ_idx] = (targets_val[:,targ_idx]-targ_min)/(targ_max-targ_min) - 0.5
    print("\nMin-Max Scaler to target '{}', from (min,max)=({:.4f},{:.4f}) to (0,1)".format(targ_name, targ_min, targ_max))
    print(f"Min-Max after scaling,  training  dataset: ({targets_tr[:, targ_idx].min():.4f},{targets_tr[:, targ_idx].max():.4f})")
    print(f"Min-Max after scaling, validation dataset: ({targets_val[:,targ_idx].min():.4f},{targets_val[:,targ_idx].max():.4f})")

# Training and validation datasets
features_tr_tf  = tf.convert_to_tensor(features_tr,  dtype=np.float32)
targets_tr_tf   = tf.convert_to_tensor(targets_tr,   dtype=np.float32)
features_val_tf = tf.convert_to_tensor(features_val, dtype=np.float32)
targets_val_tf  = tf.convert_to_tensor(targets_val,  dtype=np.float32)
dataset_tr  = tf.data.Dataset.from_tensor_slices((features_tr_tf, targets_tr_tf))
dataset_tr  = dataset_tr.shuffle(buffer_size=np.product(args.spatial_dimension)).batch(args.batch_size)
dataset_val = tf.data.Dataset.from_tensor_slices((features_val_tf, targets_val_tf))
dataset_val = dataset_val.batch(args.batch_size)
print(f"\nDataset Training: \n{dataset_tr}")
print(f"\nDataset Validation: \n{dataset_val}")

"""
# Squeeze targets, if only 1 target:
if args.num_targets == 1: 
    # squeeze:
    targets_tr  = targets_tr.reshape(-1)
    targets_val = targets_val.reshape(-1)
"""

# Loss function
class MSELoss(tf.keras.losses.Loss):
    def call(self, y_gt, y_pred):
        return tf.reduce_mean(tf.square(y_gt - y_pred), axis = 0)

class RSELoss(tf.keras.losses.Loss):
    def call(self, y_gt, y_pred):
        return tf.reduce_mean(tf.square(y_gt - y_pred)/tf.square(y_gt), axis = 0)

# MSE
if args.loss == "MSE":
    loss_func = MSELoss()
elif args.loss == "RSE":
    loss_func = RSELoss()
else:
    sys.exit("only MSE loss implemented, set argument loss to 'MSE'")

# Model: Multi-Layer Perceptron

class MLP(models.Model):
    def __init__(self, model, optimizer, loss_func, sopt=None, epochs=100, **kwargs):
        super(MLP, self).__init__(**kwargs)
        self.model = model # neural network 
        self.optimizer = optimizer # Adam optimizer
        self.sopt = sopt # L-BFGS-B optimizer 
        self.epochs = epochs # number of epochs for training using Adam
        self.hist = []
        self.hit_test = []
        self.epoch = 0
        self.loss_func = loss_func

    @tf.function
    def train_step(self, x, y_gt):
        # x:    shape [args.batch_size, args.num_features]
        # y_gt: shape [args.batch_size, args.num_targets]
        with tf.GradientTape() as tape:
            y_pred = self.model(x) 
            loss   = self.loss_func(y_gt, y_pred)
        trainable_vars = self.trainable_variables
        grads = tape.gradient(loss, trainable_vars)
        # tf.print('\nloss:', loss, summarize=-1)
        return loss, grads

    @tf.function
    def test_step(self, x, y_gt):
        y_pred = self.model(x)
        loss   = self.loss_func(y_gt, y_pred)
        # tf.print('\nloss test:', loss, summarize=-1)
        return y_pred, loss
    
    def fit(self, dataset):        
        # --> training using Adam optimizer 
        for epoch in range(1,self.epochs+1):
            loss_epoch = 0
            tf.print("\n-----------------------------------------------------------------------------")
            tf.print('Training Epoch:', self.epoch)
            for nbatch, (x_batch_tr, y_batch_tr) in enumerate(dataset):
                loss_batch, grads = self.train_step(x_batch_tr, y_batch_tr)
                loss_epoch += loss_batch
                if nbatch % 1000 == 0:
                    tf.print('batch:',nbatch,'loss batch:',loss_batch,summarize=-1)
            loss_epoch *= 1/nbatch
            tf.print('\nTraining Epoch:',epoch,'Training Loss',loss_epoch)
            self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
            self.epoch += 1
            self.hist.append(loss_epoch)   
        # --> training using L-BFGS-B optimizer
        # objective function for Scipy (L-BFGS-B) optimizer
        # # def func(params_1d):
        # #     self.sopt.assign_params(params_1d)
        # #     tf.print('epoch:', self.epoch)
        # #     loss, grads = self.train_step(x, y_gt)
        # #     grads = tf.dynamic_stitch(self.sopt.idx, grads)
        # #     self.epoch += 1
        # #     self.hist.append(loss)
        # #     return loss.numpy().astype(np.float64), grads.numpy().astype(np.float64)
        # # self.sopt.minimize(func)
        return self.hist
    
    def predict(self, dataset, args):
        loss = 0
        num_points = 0
        num_points_batch = 0
        y_pred = np.zeros(shape = args.targets_val_shape)
        tf.print("\n-----------------------------------------------------------------------------")
        tf.print("Validation Epoch")
        for nbatch, (x_batch, y_batch) in enumerate(dataset):
            y_pred_batch, loss_batch = self.test_step(x_batch, y_batch)
            loss += loss_batch
            num_points_batch = y_batch.shape[0]
            y_pred[num_points:num_points+num_points_batch,:] = y_pred_batch
            num_points += num_points_batch
            if nbatch % 1000 == 0:
                tf.print('batch:', nbatch, 'loss batch:', loss_batch, summarize=-1)
        loss *= 1/nbatch
        tf.print("\nValidation Loss:",loss)
        return y_pred, loss

    def fit_and_validate(self, dataset_tr, dataset_val, y_gt, args):
        # --> training using Adam optimizer, validate at each epoch and visualize validation results
        for epoch in range(1,self.epochs+1):
            # train
            loss_epoch = 0
            tf.print("\n-----------------------------------------------------------------------------")
            tf.print('Training Epoch:', self.epoch)
            for nbatch, (x_batch_tr, y_batch_tr) in enumerate(dataset_tr):
                loss_batch, grads = self.train_step(x_batch_tr, y_batch_tr)
                loss_epoch += loss_batch
                if nbatch < 1000 or nbatch % 1000 == 0:
                    tf.print('batch:',nbatch,'loss batch:',loss_batch,summarize=-1)
            tf.print('loss_epoch before dividing:',loss_epoch,'nbatch:',nbatch,'loss_epoch after dividing:',loss_epoch / nbatch)
            loss_epoch *= 1/nbatch
            tf.print('\nTraining Epoch:',epoch,'Training Loss',loss_epoch)
            self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
            self.epoch += 1
            self.hist.append(loss_epoch)
            # validate
            loss_val = 0
            num_points = 0
            num_points_batch = 0
            y_pred = np.zeros(shape = args.targets_val_shape)
            tf.print("\n-----------------------------------------------------------------------------")
            tf.print("Validation Epoch")
            for nbatch, (x_batch, y_batch) in enumerate(dataset_val):
                y_pred_batch, loss_batch = self.test_step(x_batch, y_batch)
                loss_val += loss_batch
                num_points_batch = y_batch.shape[0]
                y_pred[num_points:num_points+num_points_batch,:] = y_pred_batch
                num_points += num_points_batch
                if nbatch < 1000 or nbatch % 1000 == 0:
                    tf.print('batch:', nbatch, 'loss batch:', loss_batch, summarize=-1)
            loss_val *= 1/nbatch
            tf.print("\nValidation Loss:",loss_val)
            visualize_prediction(y_gt, y_pred, epoch)




act = args.activation_function
inp = layers.Input(shape = (args.num_features))
hl = inp
for i in range(args.num_hidden_layers):
    hl = layers.Dense(args.num_neurons_per_layer, activation = act)(hl)
out = layers.Dense(args.num_targets)(hl)

model = models.Model(inp, out)
print(model.summary())

# OPTIMIZER
lr = 1e-3
opt = optimizers.Adam(lr)
# sopt = SciOP(model)

# TRAINING
mlp = MLP(model, opt, loss_func=loss_func, epochs=args.num_epochs) 
# hist = mlp.fit(dataset_tr)

# VALIDATION
# targets_val_pred, loss_validation = mlp.predict(dataset_val,args=args)

mlp.fit_and_validate(dataset_tr, dataset_val, targets_val, args=args)

"""

# Model
afun = args.activation_function
npl  = args.num_neurons_per_layer
model = models.Sequential([
  layers.Flatten(input_shape=(args.num_features, 1)),
  layers.Dense(npl, activation=afun),
  layers.Dense(npl, activation=afun),
  layers.Dense(npl, activation=afun),
  layers.Dense(npl, activation=afun),
  layers.Dense(npl, activation=afun),
  layers.Dense(npl, activation=afun),
  layers.Dense(npl, activation=afun),
  layers.Dense(npl, activation=afun),
  layers.Dense(npl, activation=afun),
  layers.Dense(npl, activation=afun),
  layers.Dense(npl, activation=afun),
  layers.Dense(npl, activation=afun),
  layers.Dense(npl, activation=afun),
  layers.Dense(npl, activation=afun),
  layers.Dense(npl, activation=afun),
  layers.Dense(npl, activation=afun),
  layers.Dense(npl, activation=afun),
  layers.Dense(npl, activation=afun),
  layers.Dense(npl, activation=afun),
  layers.Dense(npl, activation=afun),
  layers.Dense(npl, activation=afun),
  layers.Dense(npl, activation=afun),
  layers.Dense(npl, activation=afun),
  layers.Dense(npl, activation=afun),
  layers.Dense(npl, activation=afun),
  layers.Dense(npl, activation=afun),
  layers.Dense(npl, activation=afun),
  layers.Dense(npl, activation=afun),
  layers.Dense(npl, activation=afun),
  layers.Dense(npl, activation=afun),
  layers.Dense(npl, activation=afun),
  layers.Dense(npl, activation=afun),
  layers.Dense(1, activation=afun)
])
optimizer = optimizers.SGD(learning_rate=args.learning_rate)
model.compile(optimizer = optimizer, loss = 'mse')
print(model.summary())

# Training
hist = model.fit(features_tr, targets_tr, epochs=15, validation_split=0.1)
targets_val_pred = model.predict(targets_val)

"""


