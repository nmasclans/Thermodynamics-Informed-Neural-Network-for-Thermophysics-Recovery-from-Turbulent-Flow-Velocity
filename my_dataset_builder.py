import numpy as np
import tensorflow as tf
import sys

def get_datasets(args):

    assert args.targets_name == ['c_p', 'rho', 'T'], "Incorrect targets_name list, code implemented for --targets_name = ['c_p','rho','T']"
    args.features_name = []
    num_snapshots_tr    = len(args.training_filename)
    num_snapshots_val   = len(args.validation_filename)

    # ----- Import Data from One or Multiple Snapshots -----

    features_tr  = np.zeros(shape = [num_snapshots_tr,]  + args.spatial_dimension + [args.num_features,], dtype=np.float32)
    features_val = np.zeros(shape = [num_snapshots_val,] + args.spatial_dimension + [args.num_features,], dtype=np.float32)
    targets_tr   = np.zeros(shape = [num_snapshots_tr,]  + args.spatial_dimension + [args.num_targets,],  dtype=np.float32)
    targets_val  = np.zeros(shape = [num_snapshots_val,] + args.spatial_dimension + [args.num_targets,],  dtype=np.float32)
    
    # Training data:
    for i_file in range(num_snapshots_tr):
        with np.load(args.training_filename[i_file]) as f:
            features_data  = f['x']
            all_features_name = f['features_names']
            for ii in range(args.num_features):
                features_tr[i_file,:,:,:,ii] = features_data[:,:,:,args.features_idx[ii]]
                args.features_name.append(all_features_name[args.features_idx[ii]])
            for tt in range(args.num_targets):
                targets_tr[i_file,:,:,:,tt]  = f[args.targets_name[tt]]

    # Validation data:
    for i_file in range(num_snapshots_val):
        with np.load(args.validation_filename[i_file]) as f:
            features_data  = f['x']
            # features_names = f['features_names']
            for ii in range(args.num_features):
                features_val[i_file,:,:,:,ii] = features_data[:,:,:,args.features_idx[ii]]
            for tt in range(args.num_targets):
                targets_val[i_file,:,:,:,tt]  = f[args.targets_name[tt]]

    # Reshape, to get one discretized node as NN input (Scalar Input):
    features_tr  = features_tr.reshape(-1,  args.num_features)
    features_val = features_val.reshape(-1, args.num_features)
    targets_tr   = targets_tr.reshape(-1,   args.num_targets)
    targets_val  = targets_val.reshape(-1,  args.num_targets)
    args.num_batches_per_epoch = int(features_tr.shape[0]/args.batch_size)
    print(f"\nShape training features: {features_tr.shape}")
    print(f"Shape training targets: {targets_tr.shape}")
    print(f"\nFeatures name: {args.features_name}")
    print(f"Targets name: {args.targets_name}")
    print(f"\nTraining Number of Batches per Epoch: {args.num_batches_per_epoch}, with number of samples {features_tr.shape[0]} and batch size {args.batch_size}")

    # Normalize features and targets data
    act_fun = args.activation_function
    if act_fun == "relu":
        args.min_value = 0; args.max_value = 1
    elif act_fun == "tanh":
        args.min_value = -1; args.max_value = 1
    else:
        sys.exit(f"argument --activation_function is {act_fun}, accepted values: 'relu', 'tanh'")
    # Normalize features:
    for feat_idx in range(args.num_features):
        feat_name = args.features_name[feat_idx]
        feat_min  = args.features_limits[feat_name][0]
        feat_max  = args.features_limits[feat_name][1]
        assert (feat_max-feat_min) > 0
        features_tr[:,feat_idx]  = (features_tr[:,feat_idx]-feat_min)  / (feat_max-feat_min) * (args.max_value-args.min_value) + args.min_value
        features_val[:,feat_idx] = (features_val[:,feat_idx]-feat_min) / (feat_max-feat_min) * (args.max_value-args.min_value) + args.min_value
        print(f"\nMin-Max Scaler to feature '{feat_name}', from (min, max) = ({feat_min:.4f}, {feat_max:.4f}) to ({args.min_value}, {args.max_value})")
        print(f"Min-Max after scaling,  training  dataset: ({features_tr[:,feat_idx].min():.4f},{features_tr[:,feat_idx].max():.4f})")
        print(f"Min-Max after scaling, validation dataset: ({features_val[:,feat_idx].min():.4f},{features_val[:,feat_idx].max():.4f})")
    # Normalize targets
    for targ_idx in range(args.num_targets):
        targ_name = args.targets_name[targ_idx]
        targ_min  = args.targets_limits[targ_name][0]
        targ_max  = args.targets_limits[targ_name][1]
        assert (targ_max-targ_min) > 0
        targets_tr[:,targ_idx]  = (targets_tr[:,targ_idx]-targ_min)  / (targ_max-targ_min) * (args.max_value-args.min_value) + args.min_value
        targets_val[:,targ_idx] = (targets_val[:,targ_idx]-targ_min) / (targ_max-targ_min) * (args.max_value-args.min_value) + args.min_value
        print(f"\nMin-Max Scaler to target '{targ_name}', from (min, max) = ({targ_min:.4f}, {targ_max:.4f}) to ({args.min_value}, {args.max_value})")
        print(f"Min-Max after scaling,  training  dataset: ({targets_tr[:, targ_idx].min():.4f},{targets_tr[:, targ_idx].max():.4f})")
        print(f"Min-Max after scaling, validation dataset: ({targets_val[:,targ_idx].min():.4f},{targets_val[:,targ_idx].max():.4f})")

    # ----- Build Datasets for Training and Validation -----
    features_tr_tf  = tf.convert_to_tensor(features_tr,  dtype=np.float32)
    targets_tr_tf   = tf.convert_to_tensor(targets_tr,   dtype=np.float32)
    features_val_tf = tf.convert_to_tensor(features_val, dtype=np.float32)
    targets_val_tf  = tf.convert_to_tensor(targets_val,  dtype=np.float32)
    dataset_tr      = tf.data.Dataset.from_tensor_slices((features_tr_tf, targets_tr_tf))
    dataset_tr      = dataset_tr.shuffle(buffer_size=dataset_tr.cardinality(), seed=args.seed).batch(args.batch_size)
    dataset_val     = tf.data.Dataset.from_tensor_slices((features_val_tf, targets_val_tf))
    dataset_val     = dataset_val.batch(args.batch_size_validation)
    print(f"\nDataset Training: \n{dataset_tr}")
    print(f"\nDataset Validation: \n{dataset_val}")

    return dataset_tr, dataset_val, args


def get_datasets_prediction(args):

    assert args.targets_name == ['c_p', 'rho', 'T'], "Incorrect targets_name list, code implemented for --targets_name = ['c_p','rho','T']"
    args.features_name = []
    num_snapshots_pred = len(args.prediction_filename)

    # ----- Import Data from One or Multiple Snapshots -----

    features_pred = np.zeros(shape = [num_snapshots_pred,] + args.spatial_dimension + [args.num_features,], dtype=np.float32)
    targets_pred  = np.zeros(shape = [num_snapshots_pred,] + args.spatial_dimension + [args.num_targets,],  dtype=np.float32)
    
    # Prediction data:
    for i_file in range(num_snapshots_pred):
        with np.load(args.validation_filename[i_file]) as f:
            features_data  = f['x']
            all_features_name = f['features_names']
            for ii in range(args.num_features):
                features_pred[i_file,:,:,:,ii] = features_data[:,:,:,args.features_idx[ii]]
                args.features_name.append(all_features_name[args.features_idx[ii]])
            for tt in range(args.num_targets):
                targets_pred[i_file,:,:,:,tt]  = f[args.targets_name[tt]]

    # Reshape, to get one discretized node as NN input (Scalar Input):
    features_pred = features_pred.reshape(-1, args.num_features)
    targets_pred  = targets_pred.reshape(-1,  args.num_targets)
    args.num_batches_per_epoch = int(features_pred.shape[0]/args.batch_size_prediction)
    print(f"\nShape prediction targets: {targets_pred.shape}")
    print(f"\nFeatures name: {args.features_name}")
    print(f"\nTargets name: {args.targets_name}")
    print(f"\nPrediction Number of Batches per Epoch: {args.num_batches_per_epoch}, with number of samples {features_pred.shape[0]} and batch size {args.batch_size_prediction}")

    # Normalize features and targets data
    act_fun = args.activation_function
    if act_fun == "relu":
        args.min_value = 0; args.max_value = 1
    elif act_fun == "tanh":
        args.min_value = -1; args.max_value = 1
    else:
        sys.exit(f"argument --activation_function is {act_fun}, accepted values: 'relu', 'tanh'")
    # Normalize features:
    for feat_idx in range(args.num_features):
        feat_name = args.features_name[feat_idx]
        feat_min  = args.features_limits[feat_name][0]
        feat_max  = args.features_limits[feat_name][1]
        assert (feat_max-feat_min) > 0
        features_pred[:,feat_idx]  = (features_pred[:,feat_idx]-feat_min)  / (feat_max-feat_min) * (args.max_value-args.min_value) + args.min_value
        print(f"\nMin-Max Scaler to feature '{feat_name}', from (min, max) = ({feat_min:.4f}, {feat_max:.4f}) to ({args.min_value}, {args.max_value})")
        print(f"Min-Max after scaling, prediction dataset: ({features_pred[:,feat_idx].min():.4f},{features_pred[:,feat_idx].max():.4f})")
    # Normalize targets
    for targ_idx in range(args.num_targets):
        targ_name = args.targets_name[targ_idx]
        targ_min  = args.targets_limits[targ_name][0]
        targ_max  = args.targets_limits[targ_name][1]
        assert (targ_max-targ_min) > 0
        targets_pred[:,targ_idx]  = (targets_pred[:,targ_idx]-targ_min)  / (targ_max-targ_min) * (args.max_value-args.min_value) + args.min_value
        print(f"\nMin-Max Scaler to target '{targ_name}', from (min, max) = ({targ_min:.4f}, {targ_max:.4f}) to ({args.min_value}, {args.max_value})")
        print(f"Min-Max after scaling, prediction dataset: ({targets_pred[:, targ_idx].min():.4f},{targets_pred[:, targ_idx].max():.4f})")

    # ----- Build Datasets for Training and Validation -----
    features_pred_tf  = tf.convert_to_tensor(features_pred,  dtype=np.float32)
    targets_pred_tf   = tf.convert_to_tensor(targets_pred,   dtype=np.float32)
    dataset_pred      = tf.data.Dataset.from_tensor_slices((features_pred_tf, targets_pred_tf))
    dataset_pred      = dataset_pred.batch(args.batch_size_prediction)
    print(f"\nDataset Prediction: \n{dataset_pred}")

    return dataset_pred, args
