'''
Execution details (Hybrid Jofre cluster)
activate conda environment: 'tf-gpu'
execute by: XLA_FLAGS=--xla_gpu_cuda_data_dir=/usr/lib/cuda python3 <python_script_name>
'''
"""
NN input is a scalar in spatial dimension, num_features per single point (x,y,z), shape [,num_features]
Features used: idx 1,2: 'u', 'TKE_normalized'
"""

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

from thermodynamics.func_GasEquation import GasEquationRelativeError


parser = argparse.ArgumentParser(description="PINN_RANS_channel_flow")
parser.add_argument("--ndim", default=3, type=int, help="problem dimensions")
parser.add_argument("--features_idx", default=[1,2], type=str, help="Selected features index")
parser.add_argument("--targets_name", default=['c_p','rho','T'], type=str, help="Selected targets name")
parser.add_argument("--training_filename", default='/home/jofre/Students/Nuria_Masclans/datasets/post_processed/59300000_5features_4targets/3d_high_pressure_turbulent_channel_flow_53900000.npz', type=str, help="List of training filenames (abspath)")
parser.add_argument("--validation_filename", default='/home/jofre/Students/Nuria_Masclans/datasets/post_processed/59300000_5features_4targets/3d_high_pressure_turbulent_channel_flow_59300000.npz', type=str, help="List of validation filenames (abspath)")
parser.add_argument("--spatial_dimension", default=[128,128,128], type=list, help="Spatial discretization, grid of statistics data. Equals the shape of the stored quantities in 'statistic")
parser.add_argument("--learning_rate", default=1e-3, type=float, help="Learning rate parameter of optimizer")
parser.add_argument("--loss", default="Supervised_PINNS", type=str, help="Loss function name, accepted values = ['MSE','RSE','Supervised_PINNS']")
parser.add_argument("--metrics", default=["MSE","RSE","RAE","Supervised_PINNS","RE_RealGasEq","RE_CpEq"], type=str, help="Metric function name")
parser.add_argument("--num_hidden_layers", default=3, type=int, help="Number of hidden layers of the model")
parser.add_argument("--num_neurons_per_layer", default=8, type=int, help="Number of neurons per layer of the model")
parser.add_argument("--activation_function", default="relu", type=str, help="Activation function (relu, tanh)")
parser.add_argument("--initializer_type", default="normal", type=str, help="Type of initializers (He, Glorot), which can be 'uniform' or 'normal'")
parser.add_argument("--initializer_seed", default=1, type=int, help="Seed of the deterministic initializer")
parser.add_argument("--num_epochs", default=100, type=int, help="Number of training epochs")
parser.add_argument("--batch_size", default=64, type=int, help="Batch size (recomended to be multiple of 8)")
parser.add_argument("--batch_size_validation", default=128**3, type=int, help="Batch size (recomended to be multiple of 8)")
parser.add_argument("--features_limits", 
    default={'y':[0.0,0.0002],'u':[0.0,3.5],'TKE_normalized':[0.0,0.2],'vorticity_magn_normalized':[0.0,55.0],'enstrophy_normalized':[0.0,1400.0]},
    type=dict, help="features minimum and maximum values, used for data normalization"
)
parser.add_argument("--targets_limits", 
    #default={'c_p':[1500.0,3900.0],'rho':[145.0,850.0],'T':[90.0,200.0]},
    default={'c_p':[0.0,3900.0],'rho':[0.0,850.0],'T':[0.0,200.0]},
    type=dict, help="targets minimum and maximum values, used for data normalization"
)
parser.add_argument("--num_batches_per_print_information", default=1000, type=int, help="number of batches for when results information is printed")
parser.add_argument("--Substance", default="N2", type=str, help="'Substance' param of thermodynamics function, relative to gas type; admissible values: 'N2'")
parser.add_argument("--P_constant", default=6791600.0, type=float, help="Pressure value [Pa], constant along all domain (2*Pc of N2)")
parser.add_argument("--visualization_step", default=100, type=int,   help="Plots data step, for less heavy plots") # TODO


args = parser.parse_args()
args.num_features = len(args.features_idx)
args.num_targets  = len(args.targets_name)
print(f"\nArguments:\n{args}")

assert args.targets_name == ['c_p', 'rho', 'T'], "Incorrect targets_name list, code implemented for --targets_name = ['c_p','rho','T']"

act_fun = args.activation_function
in_type = args.initializer_type 
in_seed = args.initializer_seed
assert act_fun in ["relu","tanh"],      f"argument --activation_function is {act_fun}, accepted values: 'relu', 'tanh'"
assert in_type in ["uniform","normal"], f"argument --initializer_type is {in_type}, accepted values: 'uniform', 'random'"
print(f"\nActivation function: {act_fun} \nInitializer type: {in_type}")

# Results Visualizaton
# We plot the quantities at the center of the computational domain
def visualize_prediction(y_gt, y_pred, epoch, args):
    palette = sns.color_palette("muted")
    for target_idx in range(args.num_targets):
        y_gt_0   = y_gt[:,target_idx]
        y_pred_0 = y_pred[:,target_idx]
        target_name = args.targets_name[target_idx]
        # probability distribution
        fig_title = f"{target_name}_kde_epoch_{epoch}"
        plt.figure()
        sns.histplot(data=y_gt_0,color=palette[0],bins=100,label="ground-truth")
        sns.histplot(data=y_pred_0,color=palette[1],bins=100,label="prediction")
        plt.xlim([args.min_value, args.max_value])
        plt.ylabel(f"{target_name} kde, density of probability")
        plt.legend()
        plt.savefig(fig_title)
        plt.close()
        print(f"Visualization of validation results in '{fig_title}'")
        # scatter plot
        fig_title = f"{target_name}_scatterplot_epoch_{epoch}"
        plt.figure()
        plt.scatter(x=y_gt_0[::args.visualization_step],y=y_pred_0[::args.visualization_step],s=1)
        plt.xlabel(f'scaled {target_name} (ground truth)')
        plt.ylabel(f'scaled {target_name} (prediction)')
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
print(f"Training Number of Batches per Epoch: {int(features_tr.shape[0]/args.batch_size)}, with number of samples {features_tr.shape[0]} and batch size {args.batch_size}")

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
dataset_val     = dataset_val.batch(args.batch_size_validation)
print(f"\nDataset Training: \n{dataset_tr}")
print(f"\nDataset Validation: \n{dataset_val}")


# Loss function

class Supervised_PINNS(tf.keras.losses.Loss):
    def __init__(self, args):
        super().__init__(name="Supervised_PINNS")
        self.supervised = RAE()
        self.Loss_PINNS_1 = RelError_RealGasEquation(args)
        self.Loss_PINNS_2 = RelError_CpEquation(args)
        self.loss_weights = [0.8, 0.1, 0.1]
        assert sum(self.loss_weights) == 1.0
    def call(self, y_gt, y_pred):
        loss_supervised = self.supervised(y_gt, y_pred)
        loss_pinns_1 = self.Loss_PINNS_1(y_gt, y_pred)
        loss_pinns_2 = self.Loss_PINNS_2(y_gt, y_pred)
        loss_supervised_PINNS = self.loss_weights[0]*loss_supervised + self.loss_weights[1]*loss_pinns_1 + self.loss_weights[2]*loss_pinns_2
        # tf.print('loss_supervised:',loss_supervised,'loss_pinns_1:',loss_pinns_1,'loss_pinns_2:',loss_pinns_2)
        # tf.print('loss_supervised_PINNS',loss_supervised_PINNS)
        return loss_supervised_PINNS

class MSE(tf.keras.losses.Loss):
    def call(self, y_gt, y_pred):
        return tf.reduce_mean(tf.square(y_gt - y_pred), axis = 0)

class RSE(tf.keras.losses.Loss):
    def call(self, y_gt, y_pred):
        y_gt += 1e-9
        return tf.reduce_mean(tf.square(y_gt - y_pred)/tf.square(y_gt), axis = 0)

class RAE(tf.keras.losses.Loss):
    def call(self, y_gt, y_pred):
        y_gt += 1e-9
        return tf.reduce_mean(tf.math.abs((y_gt - y_pred)/y_gt), axis = 0)

class RelError_RealGasEquation(tf.keras.losses.Loss):
    def __init__(self, args):
        super().__init__(name="RE_RealGasEq")
        self.rho_min   = args.targets_limits['rho'][0]
        self.rho_max   = args.targets_limits['rho'][1] 
        self.T_min     = args.targets_limits['T'][0]
        self.T_max     = args.targets_limits['T'][1] 
        self.min_value = args.min_value
        self.max_value = args.max_value
        if args.Substance == 'N2':
            self.args  = args
            # ----------------- N2 -
            self.Ru    = 8.314            # R universal
            self.MW    = 2.80134e-2       # Molecular weight kg/mol
            self.R     = self.Ru/self.MW  # R specific
            self.Tc    = 126.19           # Critical temperature [k]
            self.pc    = 3.3958e+6;       # Critical pressure [Pa]
            self.omega = 0.03720          # Acentric factor
            self.NASA_coefficients =  [ 2.952576370000000000000, 0.001396900400000000000, -0.000000492631603000000, 0.000000000078601019000,
                                       -0.000000000000004607552, -923.9486880000000000000, 5.871887620000000000000, 3.531005280000000000000,
                                       -0.000123660980000000000, -0.000000502999433000000, 0.000000002435306120000, -0.000000000001408812400,
                                       -1046.976280000000000000, 2.967470380000000000000, 0.000000000000000000000]
            if self.omega > 0.49:         # Accentric factor
                self.c = 0.379642 + 1.48503*self.omega - 0.164423*(self.omega**2) + 0.016666*(self.omega**3)
            else:
                self.c = 0.37464 + 1.54226*self.omega - 0.26992*(self.omega**2)
            self.b = 0.077796*self.R*self.Tc/self.pc
            self.P_constant = args.P_constant
        else:
            sys.exit("Not implemented Substance. Set Substance to 'N2'")

    def call(self, y_gt, y_pred):
        rho_scaled = y_pred[:,1]
        T_scaled   = y_pred[:,2]    
        rho = (rho_scaled - self.min_value) * (self.rho_max - self.rho_min) / (self.max_value - self.min_value) - self.rho_min
        T   = (T_scaled   - self.min_value) * (self.T_max   - self.T_min)   / (self.max_value - self.min_value) - self.T_min
        rho += 1e-9 # in case rho = 0, prevent v = inf, rel_err = nan
        T   += 1e-9 # in case T   = 0, prevent dadT = inf, rel_err = nan
        v    = 1/rho
        
        # -------------------------- Peng Robinson -------------------------
        a      = (0.457236*(self.R*self.Tc)**2/self.pc) * tf.pow(1+self.c*(1-tf.math.sqrt(T/self.Tc)), 2)

        # ----------------------- Equation of Real Gas ----------------------
        P = self.R * T / (v - self.b) - a / (tf.math.pow(v,2) + 2*self.b*v - self.b**2)
        
        # Relative error on the Equation of Real Gas:
        rel_err_real_gas = tf.reduce_mean(tf.math.abs((P-self.P_constant)/self.P_constant))

        return rel_err_real_gas

class RelError_CpEquation(tf.keras.losses.Loss):
    def __init__(self, args):
        super().__init__(name="RE_CpEquation")
        self.c_p_min   = args.targets_limits['c_p'][0]
        self.c_p_max   = args.targets_limits['c_p'][1] 
        self.rho_min   = args.targets_limits['rho'][0]
        self.rho_max   = args.targets_limits['rho'][1] 
        self.T_min     = args.targets_limits['T'][0]
        self.T_max     = args.targets_limits['T'][1] 
        self.min_value = args.min_value
        self.max_value = args.max_value
        if args.Substance == 'N2':
            self.args  = args
            # ----------------- N2 -
            self.Ru    = 8.314            # R universal
            self.MW    = 2.80134e-2       # Molecular weight kg/mol
            self.R     = self.Ru/self.MW  # R specific
            self.Tc    = 126.19           # Critical temperature [k]
            self.pc    = 3.3958e+6;       # Critical pressure [Pa]
            self.omega = 0.03720          # Acentric factor
            self.NASA_coefficients =  [ 2.952576370000000000000, 0.001396900400000000000, -0.000000492631603000000, 0.000000000078601019000,
                                       -0.000000000000004607552, -923.9486880000000000000, 5.871887620000000000000, 3.531005280000000000000,
                                       -0.000123660980000000000, -0.000000502999433000000, 0.000000002435306120000, -0.000000000001408812400,
                                       -1046.976280000000000000, 2.967470380000000000000, 0.000000000000000000000]
            if self.omega > 0.49:         # Accentric factor
                self.c = 0.379642 + 1.48503*self.omega - 0.164423*(self.omega**2) + 0.016666*(self.omega**3)
            else:
                self.c = 0.37464 + 1.54226*self.omega - 0.26992*(self.omega**2)
            self.b = 0.077796*self.R*self.Tc/self.pc
            self.P_constant = args.P_constant
        else:
            sys.exit("Not implemented Substance. Set Substance to 'N2'")

    def call(self, y_gt, y_pred):
        c_p_scaled = y_pred[:,0]
        rho_scaled = y_pred[:,1]
        T_scaled   = y_pred[:,2]    
        c_p = (c_p_scaled - self.min_value) * (self.c_p_max - self.c_p_min) / (self.max_value - self.min_value) - self.c_p_min
        rho = (rho_scaled - self.min_value) * (self.rho_max - self.rho_min) / (self.max_value - self.min_value) - self.rho_min
        T   = (T_scaled   - self.min_value) * (self.T_max   - self.T_min)   / (self.max_value - self.min_value) - self.T_min
        rho += 1e-9 # in case rho = 0, prevent v = inf, rel_err = nan
        T   += 1e-9 # in case T   = 0, prevent dadT = inf, rel_err = nan
        c_p += 1e-9
        v   = 1/rho

        # tf.print('c_p',c_p,summarize=-1)
        # tf.print('rho',rho,summarize=-1)
        # tf.print('T',T,summarize=-1)
        
        # -------------------------- Peng Robinson -------------------------
        a      = (0.457236*(self.R*self.Tc)**2/self.pc) * tf.pow(1+self.c*(1-tf.math.sqrt(T/self.Tc)), 2)
        G      = self.c*tf.math.sqrt(T/self.Tc) / (1+self.c*(1-tf.math.sqrt(T/self.Tc)))
        dadT   = -(1/T)*a*G
        d2adT2 = 0.457236*self.R**2 / T / 2 * self.c * (1+self.c) * self.Tc / self.pc * tf.math.sqrt(self.Tc/T)

        # --------------- Equation of  Specific Heat Capacity ---------------
        # Cp ideal gas, depending on temperature, from equation C.25, with variations because of coefficients dependency on temperature!
        #assert tf.math.reduce_all(tf.math.less_equal(T,200),T)
        # if T >= 200 and T < 1000:
        #     c_p_ideal = R*(self.NASA_coefficients[7] + self.NASA_coefficients[8]*T + self.NASA_coefficients[9]*T**2 + self.NASA_coefficients[10]*T**3 + self.NASA_coefficients[11]*T**4)
        # elif T >= 1000 and T < 6000:
        #     c_p_ideal = R*(self.NASA_coefficients[0] + self.NASA_coefficients[1]*T + self.NASA_coefficients[2]*T**2 + self.NASA_coefficients[3]*T**3 + self.NASA_coefficients[4]*T**4)
        # elif T < 200:
        #    # Assume constant temperature below 200K
        c_p_ideal = self.R*(self.NASA_coefficients[7] + self.NASA_coefficients[8]*200 + self.NASA_coefficients[9]*200**2 + self.NASA_coefficients[10]*200**3 + self.NASA_coefficients[11]*200**4)
        # else:
        #     sys.exit(f"Temperature T = {T:.3e} is too large, should be < 6000K.") 

        # Departure function Cp --> from Jofre and Urzay, appendix C.7
        Z       = self.P_constant*v/(self.R*T)          # Compressibility factor, specific
        A       = a*self.P_constant/tf.math.pow(self.R*T,2) 
        B       = self.b*self.P_constant/(self.R*T)
        M       = (tf.math.pow(Z,2) + 2*B*Z - tf.math.pow(B,2))/(Z - B)
        N       = dadT*B/(self.b*self.R)
        sqrt_2  = tf.math.sqrt(2.0)
        dep_c_p = (self.R*tf.math.pow(M - N,2))/(tf.math.pow(M,2) - 2*A*(Z + B)) - (T*d2adT2/(2*sqrt_2*self.b))*tf.math.log((Z + (1 - sqrt_2)*B)/(Z + (1 + sqrt_2)*B)) - self.R

        # Cp real gas
        c_p_equation = c_p_ideal + dep_c_p
        # Relatife error between predicted c_p vs Real Gas c_p Equation (depending on T)
        rel_err_c_p_equation = tf.reduce_mean(tf.math.abs((c_p-c_p_equation)/c_p_equation))
        return rel_err_c_p_equation

# Loss
if args.loss == "MSE":
    loss_func = MSE()
elif args.loss == "RSE":
    loss_func = RSE()
elif args.loss == "Supervised_PINNS":
    loss_func = Supervised_PINNS(args)
else:
    sys.exit(f"ValueError: Incorrect argument --loss = '{args.loss}'")
metric_func = []
for m in args.metrics:
    if m == "MSE":
        metric_func.append(MSE())
    elif m == "RAE":
        metric_func.append(RAE())
    elif m == "RSE":
        metric_func.append(RSE())
    elif m == "RE_RealGasEq":
        metric_func.append(RelError_RealGasEquation(args))
    elif m == "RE_CpEq":
        metric_func.append(RelError_CpEquation(args))
    elif m == "Supervised_PINNS":
        metric_func.append(Supervised_PINNS(args))
    else:
        sys.exit(f"ValueError: Incorrect argument --metric = '{m}'")
args.num_metrics = len(metric_func)


# Model: Multi-Layer Perceptron

class MLP(models.Model):
    def __init__(self, model, optimizer, loss_func, metric_func=[], sopt=None, epochs=100, **kwargs):
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
        metric = []
        with tf.GradientTape() as tape:
            y_pred = self.model(x) 
            loss   = self.loss_func(y_gt, y_pred)
        trainable_vars = self.trainable_variables
        grads = tape.gradient(loss, trainable_vars)
        for f in self.metric_func:
            metric.append(f(y_gt, y_pred))
        return grads, loss, metric

    @tf.function
    def test_step(self, x, y_gt):
        metric = []
        y_pred = self.model(x)
        loss   = self.loss_func(y_gt, y_pred)
        for f in self.metric_func:
            metric.append(f(y_gt, y_pred))
        return y_pred, loss, metric

    def fit(self, dataset, args):
        # --> training using Adam optimizer, validate at each epoch and visualize validation results
        for epoch in range(1,self.epochs+1):
            # train
            tf.print("\n-----------------------------------------------------------------------------")
            tf.print("Training Epoch:",epoch)
            loss_epoch = 0.0; metric_epoch = tf.zeros(args.num_metrics)
            for nbatch, (features, targets_gt) in enumerate(dataset):
                grads, loss_batch, metric_batch = self.train_step(features, targets_gt)
                self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
                loss_epoch   += loss_batch
                metric_epoch += metric_batch
                # Print batch results, if required
                if nbatch % args.num_batches_per_print_information == 0:
                    tf.print("    Batch:", nbatch, f"|| Loss '{args.loss}':", loss_batch, f"|| Metric {args.metrics}:", metric_batch)
               # if nbatch == 100:
               #     sys.exit()
            loss_epoch *= 1/(nbatch+1); metric_epoch *= 1/(nbatch+1)
            tf.print("\nTraining Epoch:",epoch)
            tf.print(f"Loss '{args.loss}':",loss_epoch)
            tf.print(f"Metric {args.metrics}:", metric_epoch)
            self.hist.append(loss_epoch)
            # Print time
            now = datetime.now().strftime("%H:%M:%S")
            print(f"Current Time: {now}")

    def validate(self, dataset, args, epoch=0):
        loss_val = 0; metric_val = np.zeros(args.num_metrics)
        tf.print("\n-----------------------------------------------------------------------------")
        tf.print("Validation Epoch")
        for nbatch, (features, targets_gt) in enumerate(dataset):
            targets_pred, loss_batch, metric_batch = self.test_step(features, targets_gt)
            visualize_prediction(targets_gt, targets_pred, epoch, args)
            loss_val += loss_batch; metric_val += metric_batch
        loss_val *= 1/(nbatch+1); metric_val *= 1/(nbatch+1)
        tf.print("\nValidation Epoch:",epoch)
        tf.print(f"Loss '{args.loss}':",loss_val)
        tf.print(f"Metric {args.metrics}:", metric_val)
        # print time
        now = datetime.now().strftime("%H:%M:%S")
        tf.print("Current Time:",now)

    def fit_and_validate(self, dataset_tr, dataset_val, args):
            # --> training using Adam optimizer, validate at each epoch and visualize validation results
        for epoch in range(1,self.epochs+1):
            # train
            tf.print("\n-----------------------------------------------------------------------------")
            tf.print("Training Epoch:",epoch)
            loss_epoch = 0.0; metric_epoch = tf.zeros(args.num_metrics)
            for nbatch, (features, targets_gt) in enumerate(dataset_tr):
                grads, loss_batch, metric_batch = self.train_step(features, targets_gt)
                self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
                loss_epoch   += loss_batch
                metric_epoch += metric_batch
                # Print batch results, if required
                if nbatch % args.num_batches_per_print_information == 0:
                    tf.print("    Batch:", nbatch, f"|| Loss '{args.loss}':", loss_batch, f"|| Metric {args.metrics}:", metric_batch)
               # if nbatch == 100:
               #     sys.exit()
            loss_epoch *= 1/(nbatch+1); metric_epoch *= 1/(nbatch+1)
            tf.print('\nTraining Epoch:',epoch)
            tf.print(f"Loss '{args.loss}':",loss_epoch)
            tf.print(f"Metric {args.metrics}:", metric_epoch)
            self.hist.append(loss_epoch)
            # Print time
            now = datetime.now().strftime("%H:%M:%S")
            print(f"Current Time: {now}")
            # Validation, make plots
            self.validate(dataset_val, args, epoch)


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
out = layers.Dense(args.num_targets, activation = act_fun, kernel_initializer = initializer)(hl)

model = models.Model(inp, out)
print(model.summary())

# OPTIMIZER
lr = 1e-3
opt = optimizers.Adam(lr)
# sopt = SciOP(model)

# TRAINING
mlp = MLP(model, opt, loss_func=loss_func, metric_func=metric_func, epochs=args.num_epochs) 
#mlp.fit(dataset_tr, args)
mlp.fit_and_validate(dataset_tr, dataset_val, args)
