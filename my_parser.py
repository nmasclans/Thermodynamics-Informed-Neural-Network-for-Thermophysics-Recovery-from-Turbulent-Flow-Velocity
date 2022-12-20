import argparse


def get_arguments():

    parser = argparse.ArgumentParser(description="PINN_RANS_channel_flow")
    parser.add_argument("--ndim", default=3, type=int, help="problem dimensions")
    parser.add_argument("--features_idx", default=[1,2], type=str, help="Selected features index")
    parser.add_argument("--targets_name", default=['c_p','rho','T'], type=str, help="Selected targets name")
    parser.add_argument("--training_filename", default='/home/jofre/Students/Nuria_Masclans/datasets/post_processed/59300000_5features_4targets/3d_high_pressure_turbulent_channel_flow_53900000.npz', type=str, help="List of training filenames (abspath)")
    parser.add_argument("--validation_filename", default='/home/jofre/Students/Nuria_Masclans/datasets/post_processed/59300000_5features_4targets/3d_high_pressure_turbulent_channel_flow_59300000.npz', type=str, help="List of validation filenames (abspath)")
    parser.add_argument("--spatial_dimension", default=[128,128,128], type=list, help="Spatial discretization, grid of statistics data. Equals the shape of the stored quantities in 'statistic")
    parser.add_argument("--learning_rate", default=1e-4, type=float, help="Learning rate parameter of optimizer")
    parser.add_argument("--optimizer", default="Adam", type=str, help="Optimizer algorithm, Optimizer name")
    parser.add_argument("--loss", default="Supervised_PINNS", type=str, help="Loss function name, accepted values = ['MSE','RSE','Supervised_PINNS']")
    parser.add_argument("--metrics", default=["MSE","RSE","RAE","Supervised_PINNS","RE_RealGasEq","RE_CpEq"], type=str, help="Metric function name")
    parser.add_argument("--Supervised_PINNS_weights", default="0.8, 0.1, 0.1", type=str, help="Weights of Supervised_PINNS Loss")
    parser.add_argument("--num_hidden_layers", default=3, type=int, help="Number of hidden layers of the model")
    parser.add_argument("--num_neurons_per_layer", default=8, type=int, help="Number of neurons per layer of the model")
    parser.add_argument("--activation_function", default="relu", type=str, help="Activation function (relu, tanh)")
    parser.add_argument("--initializer_type", default="normal", type=str, help="Type of initializers (He, Glorot), which can be 'uniform' or 'normal'")
    parser.add_argument("--seed", default=1, type=int, help="Seed of the deterministic randomizers")
    parser.add_argument("--regularizer_type", default="None", type=str, help="Layers regularizer, accepted values: 'None', 'L1', 'L2'")
    parser.add_argument("--regularizer_factor", default=1e-3, type=float, help="Regularizer factor")
    parser.add_argument("--num_epochs", default=100, type=int, help="Number of training epochs")
    parser.add_argument("--batch_size", default=64, type=int, help="Batch size (recomended to be multiple of 8)")
    parser.add_argument("--batch_size_validation", default=128**3, type=int, help="Batch size (recomended to be multiple of 8)")
    parser.add_argument("--features_limits", 
        default={'y':[0.0,0.0002],'u':[0.0,3.5],'TKE_normalized':[0.0,0.2],'vorticity_magn_normalized':[0.0,55.0],'enstrophy_normalized':[0.0,1400.0]},
        type=dict, help="features minimum and maximum values, used for data normalization"
    )
    parser.add_argument("--targets_limits", 
        #default={'c_p':[1500.0,3900.0],'rho':[145.0,850.0],'T':[90.0,200.0]},
        default={'c_p':[0.0,4200.0],'rho':[0.0,850.0],'T':[0.0,200.0]},
        type=dict, help="targets minimum and maximum values, used for data normalization"
    )
    parser.add_argument("--num_batches_per_print_information", default=1000000000, type=int, help="number of batches for when results information is printed")
    parser.add_argument("--Substance", default="N2", type=str, help="'Substance' param of thermodynamics function, relative to gas type; admissible values: 'N2'")
    parser.add_argument("--P_constant", default=6791600.0, type=float, help="Pressure value [Pa], constant along all domain (2*Pc of N2)")
    parser.add_argument("--visualization_step", default=100, type=int,   help="Plots data step, for less heavy plots") # TODO
    parser.add_argument("--eps", default=1e-5, type=float, help="Epsilon, small value for preventing nan losses when output values are 0") # TODO
    parser.add_argument("--make_plots", default=False, type=bool, help="If True, make plots (scatter, histogram) in validation epoch") # TODO

    args = parser.parse_args()
    args.num_features = len(args.features_idx)
    args.num_targets  = len(args.targets_name)
    args.Supervised_PINNS_weights = list(map(float,args.Supervised_PINNS_weights.split(',')))
    
    print("\nArguments:\n" + "".join(f"\n{k}: {v}\n" for k, v in vars(args).items()))

    return args