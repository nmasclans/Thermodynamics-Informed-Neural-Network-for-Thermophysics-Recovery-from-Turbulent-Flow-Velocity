import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

# Results Visualizaton

# Plot the quantities of the validation dataset, single validation batch
# Plot scatterplot + histogram

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
        # print(f"Visualization of validation results in '{fig_title}'")
        # scatter plot
        fig_title = f"{target_name}_scatterplot_epoch_{epoch}"
        plt.figure()
        plt.scatter(x=y_gt_0[::args.visualization_step],y=y_pred_0[::args.visualization_step],s=1)
        plt.xlabel(f'scaled {target_name} (ground truth)')
        plt.ylabel(f'scaled {target_name} (prediction)')
        plt.savefig(fig_title)
        plt.close()
        # print(f"Visualization of validation results in '{fig_title}'")