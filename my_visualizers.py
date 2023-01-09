import numpy as np
from matplotlib import pyplot as plt

# Results Visualizaton

# Plot the quantities of the validation dataset, single validation batch
# Plot scatterplot + histogram

def visualize_prediction(y_gt, y_pred, epoch, batch, args):
    for target_idx in range(args.num_targets):
        y_gt_0   = y_gt[:,target_idx]
        y_pred_0 = y_pred[:,target_idx]
        target_name = args.targets_name[target_idx]
        # probability distribution
        fig_title = f"{target_name}_kde_E{epoch}_B{batch}.png"
        plt.figure()
        plt.hist(x=y_gt_0,  bins=100,label="ground-truth")
        plt.hist(x=y_pred_0,bins=100,label="prediction")
        plt.xlim([args.min_value, args.max_value])
        plt.ylabel(f"{target_name} kde, density of probability")
        plt.legend()
        plt.savefig(fig_title)
        plt.close()
        # print(f"Visualization of validation results in '{fig_title}'")
        # scatter plot
        fig_title = f"{target_name}_scatterplot_E{epoch}_B{batch}.png"
        plt.figure()
        plt.scatter(x=y_gt_0[::args.visualization_step],y=y_pred_0[::args.visualization_step],s=1)
        plt.xlabel(f'scaled {target_name} (ground truth)')
        plt.ylabel(f'scaled {target_name} (prediction)')
        plt.savefig(fig_title)
        plt.close()
        # print(f"Visualization of validation results in '{fig_title}'")

def visualize_prediction_by_xyplanes(y_gt, y_pred, epoch, batch, args):
    assert y_gt.ndim == 2 and y_pred.ndim == 2
    assert y_gt.shape[0] == 128**3 and y_pred.shape[0] == 128**3, "--batch_size of prediction must be set to 128**3"
    for target_idx in range(args.num_targets):
        y_gt_target   = y_gt[:,target_idx]
        y_pred_target = y_pred[:,target_idx]
        target_name = args.targets_name[target_idx]
        # recover spatial dimension
        y_gt_target    = y_gt_target.reshape(args.spatial_dimension)         # shape: [128,128,128]
        y_pred_target  = y_pred_target.reshape(args.spatial_dimension)
        coord_x = np.linspace(0,1,args.spatial_dimension[2])
        coord_y = np.linspace(0,1,args.spatial_dimension[1])
        # plot contourf of middle plane z
        # ground truth
        fig_title = f"{target_name}_contourf_E{epoch}_B{batch}_gt.png"
        plt.figure()
        plt.contourf(coord_y, coord_x, y_gt_target[64,:,:]); plt.axis('scaled'); plt.colorbar()
        plt.savefig(fig_title)
        # prediction
        fig_title = f"{target_name}_contourf_E{epoch}_B{batch}_pred.png"
        plt.figure()
        plt.contourf(coord_y, coord_x, y_pred_target[64,:,:]); plt.axis('scaled'); plt.colorbar()
        plt.savefig(fig_title)
