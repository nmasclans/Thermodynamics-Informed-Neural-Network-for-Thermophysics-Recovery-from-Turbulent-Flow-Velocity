'''
Execution details (Hybrid Jofre cluster)
activate conda environment: 'tf-gpu'
execute by: XLA_FLAGS=--xla_gpu_cuda_data_dir=/usr/lib/cuda python3 <python_script_name>
'''

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import argparse
import os
import sys

import numpy as np
import seaborn as sns
import tensorflow as tf
from datetime import datetime

from matplotlib import pyplot as plt
from tensorflow.keras import models, layers, optimizers, activations, initializers
# from ScipyOP import optimizer as SciOP # L-BFGS-B optimizer


parser = argparse.ArgumentParser(description="PINN_RANS_channel_flow")
parser.add_argument("--ndim", default=3, type=int, help="problem dimensions")
parser.add_argument("--features_idx", default=[1,2], type=str, help="Selected features index")
parser.add_argument("--targets_name", default=['rho',], type=str, help="Selected targets name")
parser.add_argument("--training_filename", default='/home/jofre/Students/Nuria_Masclans/datasets/post_processed/59300000_5features_4targets/3d_high_pressure_turbulent_channel_flow_59300000.npz', type=str, help="List of training filenames (abspath)")
parser.add_argument("--validation_filename", default='/home/jofre/Students/Nuria_Masclans/datasets/post_processed/59300000_5features_4targets/3d_high_pressure_turbulent_channel_flow_59300000.npz', type=str, help="List of validation filenames (abspath)")
parser.add_argument("--spatial_dimension", default=[128,128,128], type=list, help="Spatial discretization, grid of statistics data. Equals the shape of the stored quantities in 'statistic")
parser.add_argument("--learning_rate", default=1e-3, type=float, help="Learning rate parameter of optimizer")
parser.add_argument("--loss", default="MSE", type=str, help="Loss function name")
parser.add_argument("--metrics", default=["RAE",], type=str, help="Metric function name")
parser.add_argument("--num_hidden_layers", default=3, type=int, help="Number of hidden layers of the model")
parser.add_argument("--num_neurons_per_layer", default=8, type=int, help="Number of neurons per layer of the model")
parser.add_argument("--activation_function", default="relu", type=str, help="Activation function (relu, tanh)")
parser.add_argument("--initializer_type", default="uniform", type=str, help="Type of initializers (He, Glorot), which can be 'uniform' or 'normal'")
parser.add_argument("--initializer_seed", default=13, type=int, help="Seed of the deterministic initializer")
parser.add_argument("--num_epochs", default=20, type=int, help="Number of training epochs")
parser.add_argument("--batch_size", default=16, type=int, help="Batch size (recomended to be multiple of 8)")
parser.add_argument("--features_limits", 
    default={'y':[0.0,0.0002],'u':[0.0,3.5],'TKE_normalized':[0.0,0.2],'vorticity_magn_normalized':[0.0,55.0],'enstrophy_normalized':[0.0,1400.0]},
    type=dict, help="features minimum and maximum values, used for data normalization"
)
parser.add_argument("--targets_limits", 
    default={'c_p':[1500.0,3900.0],'rho':[145.0,850.0],'T':[90.0,200.0]},
    type=dict, help="targets minimum and maximum values, used for data normalization"
)
parser.add_argument("--num_batches_per_print_information", default=25000, type=int, help="number of batches for when results information is printed")

args = parser.parse_args()
args.num_features = len(args.features_idx)
args.num_targets  = len(args.targets_name)
print(f"\nArguments:\n{args}")

act_fun = args.activation_function
in_type = args.initializer_type 
in_seed = args.initializer_seed
assert act_fun in ["relu","tanh"],      f"argument --activation_function is {act_fun}, accepted values: 'relu', 'tanh'"
assert in_type in ["uniform","normal"], f"argument --initializer_type is {in_type}, accepted values: 'uniform', 'random'"

# Results Visualizaton
# We plot the quantities at the center of the computational domain
def visualize_prediction(y_gt, y_pred, epoch, args):
    y_gt_0   = y_gt[:,0]
    y_pred_0 = y_pred[:,0]
    palette = sns.color_palette("muted")
    target_idx = 0

    # probability distribution
    fig_title = f"rho_kde_epoch_{epoch}"
    plt.figure()
    sns.histplot(data=y_gt_0,color=palette[0],bins=100,label="ground-truth")
    sns.histplot(data=y_pred_0,color=palette[1],bins=100,label="prediction")
    plt.xlim([args.min_value, args.max_value])
    plt.ylabel("rho kde, density of probability")
    plt.legend()
    plt.savefig(fig_title)
    plt.close()
    print(f"Visualization of validation results in '{fig_title}'")
    # scatter plot
    fig_title = f"rho_scatterplot_epoch_{epoch}"
    plt.figure()
    plt.scatter(x=y_gt_0[::args.visualization_step],y=y_pred_0[::args.visualization_step],s=1)
    plt.xlabel('scaled rho (ground truth)')
    plt.ylabel('scaled rho (prediction)')
    plt.savefig(fig_title)
    plt.close()
    print(f"Visualization of validation results in '{fig_title}'")

# ____________________________________________________________________________
#
#           Import statistics data from args.statistics_filename
# ____________________________________________________________________________


features_tr  = np.zeros(shape = args.spatial_dimension + [args.num_features,], dtype=np.float32)
features_val = np.zeros(shape = args.spatial_dimension + [args.num_features,], dtype=np.float32)
targets_tr   = np.zeros(shape = args.spatial_dimension + [args.num_targets,],  dtype=np.float32)
targets_val  = np.zeros(shape = args.spatial_dimension + [args.num_targets,],  dtype=np.float32)
args.features_name = []
with np.load(args.training_filename) as f:
    features_data  = f['x']
    all_features_name = f['features_names']
    for ii in range(args.num_features):
        features_tr[:,:,:,ii] = features_data[:,:,:,args.features_idx[ii]]
        args.features_name.append(all_features_name[args.features_idx[ii]])
    for tt in range(args.num_targets):
        targets_tr[:,:,:,tt]  = f[args.targets_name[tt]]
with np.load(args.validation_filename) as f:
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
if act_fun == "relu":
    args.min_value = 0; args.max_value = 1
if act_fun == "tanh":
    args.min_value = -1; args.max_value = 1
for feat_idx in range(args.num_features):
    feat_name = args.features_name[feat_idx]
    feat_min  = args.features_limits[feat_name][0]
    feat_max  = args.features_limits[feat_name][1]
    assert (feat_max-feat_min) > 0
    features_tr[:,feat_idx]  = (features_tr[:,feat_idx]-feat_min)  / (feat_max-feat_min) * (args.max_value-args.min_value) + args.min_value
    features_val[:,feat_idx] = (features_val[:,feat_idx]-feat_min) / (feat_max-feat_min) * (args.max_value-args.min_value) + args.min_value
    print(f"\nMin-Max Scaler to feature '{feat_name}', from (min,max)=({feat_min:.4f},{feat_max:.4f}) to ({args.min_value}, {args.max_value})")
    print(f"Min-Max after scaling,  training  dataset: ({features_tr[:,feat_idx].min():.4f},{features_tr[:,feat_idx].max():.4f})")
    print(f"Min-Max after scaling, validation dataset: ({features_val[:,feat_idx].min():.4f},{features_val[:,feat_idx].max():.4f})")
for targ_idx in range(args.num_targets):
    targ_name = args.targets_name[targ_idx]
    targ_min  = args.targets_limits[targ_name][0]
    targ_max  = args.targets_limits[targ_name][1]
    assert (targ_max-targ_min) > 0
    targets_tr[:,targ_idx]  = (targets_tr[:,targ_idx]-targ_min)  / (targ_max-targ_min) * (args.max_value-args.min_value) + args.min_value
    targets_val[:,targ_idx] = (targets_val[:,targ_idx]-targ_min) / (targ_max-targ_min) * (args.max_value-args.min_value) + args.min_value
    print(f"\nMin-Max Scaler to target '{targ_name}', from (min,max)=({targ_min:.4f},{targ_max:.4f}) to ({args.min_value},{args.max_value})")
    print(f"Min-Max after scaling,  training  dataset: ({targets_tr[:, targ_idx].min():.4f},{targets_tr[:, targ_idx].max():.4f})")
    print(f"Min-Max after scaling, validation dataset: ({targets_val[:,targ_idx].min():.4f},{targets_val[:,targ_idx].max():.4f})")

# Training and validation datasets
features_tr_tf  = tf.convert_to_tensor(features_tr,  dtype=np.float32)
targets_tr_tf   = tf.convert_to_tensor(targets_tr,   dtype=np.float32)
features_val_tf = tf.convert_to_tensor(features_val, dtype=np.float32)
targets_val_tf  = tf.convert_to_tensor(targets_val,  dtype=np.float32)
dataset_tr      = tf.data.Dataset.from_tensor_slices((features_tr_tf, targets_tr_tf))
dataset_tr      = dataset_tr.shuffle(buffer_size=np.product(args.spatial_dimension)).batch(args.batch_size)
dataset_val     = tf.data.Dataset.from_tensor_slices((features_val_tf, targets_val_tf))
dataset_val     = dataset_val.batch(args.batch_size)
print(f"\nDataset Training: \n{dataset_tr}")
print(f"\nDataset Validation: \n{dataset_val}")


# Loss function
class MSE(tf.keras.losses.Loss):
    def call(self, y_gt, y_pred):
        return tf.reduce_mean(tf.square(y_gt - y_pred), axis = 0)

class RSE(tf.keras.losses.Loss):
    def call(self, y_gt, y_pred):
        return tf.reduce_mean(tf.square(y_gt - y_pred)/tf.square(y_gt), axis = 0)

class RAE(tf.keras.losses.Loss):
    def call(self, y_gt, y_pred):
        return tf.reduce_mean(tf.math.abs((y_gt - y_pred)/y_gt), axis = 0)

# Loss
if args.loss == "MSE":
    loss_func = MSE()
elif args.loss == "RSE":
    loss_func = RSE()
else:
    sys.exit("only MSE loss implemented, set argument loss to 'MSE'")
if "RAE" in args.metrics:
    metric_func = RAE()


# Model: Multi-Layer Perceptron

class MLP(models.Model):
    def __init__(self, model, optimizer, loss_func, metric_func=None, sopt=None, epochs=100, **kwargs):
        super(MLP, self).__init__(**kwargs)
        self.model = model # neural network 
        self.optimizer = optimizer # Adam optimizer
        self.sopt = sopt # L-BFGS-B optimizer 
        self.epochs = epochs # number of epochs for training using Adam
        self.hist = []
        self.hist_test = []
        self.epoch = 0
        self.loss_func = loss_func
        self.metric_func = metric_func

    @tf.function
    def train_step(self, x, y_gt):
        # x:    shape [args.batch_size, args.num_features]
        # y_gt: shape [args.batch_size, args.num_targets]
        metric = None
        with tf.GradientTape() as tape:
            y_pred = self.model(x) 
            loss   = self.loss_func(y_gt, y_pred)
        trainable_vars = self.trainable_variables
        grads = tape.gradient(loss, trainable_vars)
        if self.metric_func is not None:
            metric = metric_func(y_gt, y_pred)
        return grads, loss, metric

    @tf.function
    def test_step(self, x, y_gt):
        metric = None
        y_pred = self.model(x)
        loss   = self.loss_func(y_gt, y_pred)
        if self.metric_func is not None:
            metric = metric_func(y_gt, y_pred)
        return y_pred, loss, metric

    def fit(self, dataset, args):
        # --> training using Adam optimizer, validate at each epoch and visualize validation results
        for epoch in range(1,self.epochs+1):
            # train
            tf.print("\n-----------------------------------------------------------------------------")
            tf.print("Training Epoch:",epoch)
            loss_epoch = 0; metric_epoch = 0
            for nbatch, (features, targets_gt) in enumerate(dataset):
                grads, loss_batch, metric_batch = self.train_step(features, targets_gt)
                self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
                loss_epoch   += loss_batch
                metric_epoch += metric_batch
                # Print batch results, if required
                if nbatch % args.num_batches_per_print_information == 0:
                    tf.print(f"    Batch: {nbatch}, Loss {loss_batch:.5f}, Metric {metric_batch:.5f}")
            loss_epoch *= 1/nbatch; metric_epoch *= 1/nbatch
            tf.print('\nTraining Epoch:',epoch,', Loss:',loss_epoch,', Metric:',metric_epoch)
            self.hist.append(loss_epoch)
            # Print time
            now = datetime.now().strftime("%H:%M:%S")
            print(f"Current Time: {now}")


    def validate(self, dataset, args):
        loss_val = 0; metric_val = 0
        tf.print("\n-----------------------------------------------------------------------------")
        tf.print("Validation Epoch")
        for nbatch, (features, targets_gt) in enumerate(dataset):
            _, loss_batch, metric_batch = self.test_step(features, targets_gt)
            loss_val += loss_batch; metric_val += metric_batch
        loss_val *= 1/nbatch; metric_val *= 1/nbatch
        tf.print(f"\nValidation Loss: {loss_val:.5f}, Metric: {metric_val:.5f}")
        # print time
        now = datetime.now().strftime("%H:%M:%S")
        print(f"Current Time: {now}")


if act_fun == "relu":
    if in_type == "uniform":
        initializer = initializers.HeUniform(seed=in_seed)
    else: # in_type == "normal":
        initializer = initializers.HeNormal(seed=in_seed)
else: # act == 'tanh':
    if in_type == "uniform":
        initializer = initializers.GlorotUniform(seed=in_seed)
    else: # in_type  == "normal":
        initializer = initializers.GlorotNormal(seed=in_seed)

inp = layers.Input(shape = (args.num_features))
hl = inp
for i in range(args.num_hidden_layers):
    hl = layers.Dense(args.num_neurons_per_layer, activation = act_fun, kernel_initializer=initializer)(hl)
out = layers.Dense(args.num_targets, kernel_initializer = initializer)(hl)

model = models.Model(inp, out)
print(model.summary())

# OPTIMIZER
lr = 1e-3
opt = optimizers.Adam(lr)
# sopt = SciOP(model)

# TRAINING
mlp = MLP(model, opt, loss_func=loss_func, metric_func=metric_func, epochs=args.num_epochs) 
mlp.fit(dataset_tr, args)
