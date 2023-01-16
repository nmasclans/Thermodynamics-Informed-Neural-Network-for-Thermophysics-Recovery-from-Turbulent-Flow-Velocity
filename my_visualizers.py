import copy
import numpy as np
import tensorflow as tf

from matplotlib import pyplot as plt
from matplotlib import colors
from matplotlib import cm
from matplotlib.ticker import LogFormatter, FuncFormatter

from my_metrics_classification import compute_metrics
from my_utils import transform_targets_to_original_scaling, classify_state_from_temperature

# Conf. for Classification Colormap
# set colors, labels and bins
colors_dict = {0:[0.0,150/255,1.0,0.74], 1:[0.0,150/255,0.0,0.57],2:[1.0,0.0,0.0,0.54]}
labels_arr  = np.array(['liquid-like','two-phase-like','gas-like'])
labels_len  = len(labels_arr)
norm_bins   = np.sort([*colors_dict.keys()]) + 0.5
norm_bins   = np.insert(norm_bins, 0, np.min(norm_bins) - 1.0)
# set colormap, norm and formatter
cmap  = colors.ListedColormap([colors_dict[c] for c in colors_dict.keys()])
norm  = colors.BoundaryNorm(norm_bins, labels_len, clip=True)
fmt   = FuncFormatter(lambda x, pos: labels_arr[norm(x)])
# set colorbar ticks
diff  = norm_bins[1:] - norm_bins[:-1]
tickz = norm_bins[:-1] + diff/2  


# Plot the quantities of the validation dataset, single validation batch
# Plot scatterplot + histogram
def visualize_prediction(y_gt, y_pred, epoch, batch, args):
    (y_gt, y_pred) = transform_targets_to_original_scaling(args, y_gt, y_pred)
    for target_idx in range(args.num_targets):
        y_gt_0   = y_gt[:,target_idx]
        y_pred_0 = y_pred[:,target_idx]
        target_name = args.targets_name[target_idx]
        # probability distribution
        fig_title = f"figures/{target_name}_kde_E{epoch}_B{batch}.png"
        plt.figure()
        plt.hist(x=y_gt_0,  bins=100,label="ground-truth")
        plt.hist(x=y_pred_0,bins=100,label="prediction")
        plt.ylabel(f"{target_name} kde, density of probability")
        plt.legend()
        plt.savefig(fig_title)
        plt.close()
        # print(f"Visualization of validation results in '{fig_title}'")
        # scatter plot
        fig_title = f"figures/{target_name}_scatterplot_E{epoch}_B{batch}.png"
        plt.figure()
        plt.scatter(x=y_gt_0[::args.visualization_step],y=y_pred_0[::args.visualization_step],s=1)
        plt.xlabel(f'scaled {target_name} (ground truth)')
        plt.ylabel(f'scaled {target_name} (prediction)')
        plt.savefig(fig_title)
        plt.close()
        # print(f"Visualization of validation results in '{fig_title}'")

def visualize_prediction_regression_by_xyplanes(y_gt, y_pred, epoch, batch, args):
    assert y_gt.ndim == 2 and y_pred.ndim == 2
    assert y_gt.shape[0] == 128**3 and y_pred.shape[0] == 128**3, "--batch_size of prediction must be set to 128**3"
    (y_gt, y_pred) = transform_targets_to_original_scaling(args, y_gt, y_pred)
    for target_idx in range(args.num_targets):
        y_gt_target   = y_gt[:,target_idx]
        y_pred_target = y_pred[:,target_idx]
        target_name = args.targets_name[target_idx]
        # recover spatial dimension
        y_gt_target_d   = tf.reshape(y_gt_target,   args.spatial_dimension)         # shape: [128,128,128]
        y_pred_target_d = tf.reshape(y_pred_target, args.spatial_dimension)
        # --> plot imshow of middle plane z
        # ground truth
        fig_title = f"figures/{target_name}_contourf_E{epoch}_B{batch}_gt.png"
        fig_title_2 = f"figures/{target_name}_contourf_E{epoch}_B{batch}_gt.svg"
        fig = plt.figure()
        plt.imshow(y_gt_target_d[64,:,:],cmap='coolwarm'); plt.axis('scaled'); plt.colorbar()
        plt.gca().invert_yaxis()
        plt.xticks([]),plt.yticks([])
        fig.savefig(fig_title); fig.savefig(fig_title_2)
        plt.close(fig)
        # prediction
        fig_title = f"figures/{target_name}_contourf_E{epoch}_B{batch}_pred.png"
        fig_title_2 = f"figures/{target_name}_contourf_E{epoch}_B{batch}_pred.svg"
        fig = plt.figure()
        plt.imshow(y_pred_target_d[64,:,:],cmap='coolwarm'); plt.axis('scaled'); plt.colorbar()
        plt.gca().invert_yaxis()
        plt.xticks([]),plt.yticks([])
        fig.savefig(fig_title); fig.savefig(fig_title_2)
        plt.close(fig)
        # --> plot joinpdf of all fluid domain
        # y_gt_middleplane_flat   = tf.reshape(y_gt_target_d[64,:,:],-1)
        # y_pred_middleplane_flat = tf.reshape(y_pred_target_d[64,:,:],-1)
        plot_join_pdf(y_gt_target, y_pred_target, target_name, nbins = 200)


def visualize_prediction_classification_by_xyplanes(y_gt, y_pred, epoch, batch, args):
    assert y_gt.ndim == 2 and y_pred.ndim == 2
    assert y_gt.shape[0] == 128**3 and y_pred.shape[0] == 128**3, "--batch_size of prediction must be set to 128**3"
    assert args.targets_name[2] == 'T', "'visualize_prediction_classification_by_xyplanes' function needs --target_name 3rd element to be 'T'"
    (y_gt, y_pred) = transform_targets_to_original_scaling(args, y_gt, y_pred)
    T_gt   = y_gt[:,2]
    T_pred = y_pred[:,2]
    # predict class from regression data!
    (state_gt, state_pred) = classify_state_from_temperature(args, T_gt, T_pred)
    # compute (and print) metrics
    compute_metrics(state_gt, state_pred)
    # recover spatial dimension
    state_gt_d   = tf.reshape(state_gt,   args.spatial_dimension)   # shape: [128,128,128]
    state_pred_d = tf.reshape(state_pred, args.spatial_dimension)
    # plot contourf of middle plane z
    # --> ground truth
    fig_title = f"figures/fluid_state_contourf_E{epoch}_B{batch}_gt.png"
    fig_title_2 = f"figures/fluid_state_contourf_E{epoch}_B{batch}_gt.svg"
    fig, ax = plt.subplots()
    img = ax.imshow(state_gt_d[64,:,:], interpolation='nearest', origin='lower',
               cmap=cmap, norm=norm)
    fig.colorbar(img, norm=norm, format=fmt, ticks=tickz)
    ax.set_xticks([]); ax.set_yticks([])
    fig.savefig(fig_title); fig.savefig(fig_title_2)
    plt.close(fig)
    del img, fig_title, fig_title_2, fig, ax
    # --> prediction
    fig_title = f"figures/fluid_state_contourf_E{epoch}_B{batch}_pred.png"
    fig_title_2 = f"figures/fluid_state_contourf_E{epoch}_B{batch}_pred.svg"
    fig, ax = plt.subplots()
    img = ax.imshow(state_pred_d[64,:,:], interpolation='nearest', origin='lower',
               cmap=cmap, norm=norm)
    fig.colorbar(img, norm=norm, format=fmt, ticks=tickz)
    ax.set_xticks([]); ax.set_yticks([])
    fig.savefig(fig_title); fig.savefig(fig_title_2)
    plt.close(fig)


def plot_join_pdf(gt, pred, target_name, nbins = 200):
    fig, ax = plt.subplots()
    data_min = tf.reduce_min([gt, pred])
    data_max = tf.reduce_max([gt, pred])
    tf.print(f"Minimum value of {target_name} for (gt & pred) is", data_min)
    tf.print(f"Maximum value of {target_name} for (gt & pred) is", data_max)
    # Histogram 2D plot
    x_bins = np.linspace( data_min, data_max, nbins )
    y_bins = np.linspace( data_min, data_max, nbins )
    h, x_edges, y_edges = np.histogram2d( gt, pred, bins = [ x_bins, y_bins ], normed = True )
    h = h + 1.0e-12
    h = h.T
    x_centers = ( x_edges[:-1] + x_edges[1:] )/2
    y_centers = ( y_edges[:-1] + y_edges[1:] )/2
    # Plot data
    # plt.clf()
    #my_cmap = copy.copy( cm.get_cmap( 'Greys' ) )
    my_cmap = copy.copy( cm.get_cmap( 'pink_r' ) )
    my_cmap.set_under( 'white' )
    cs = ax.contour( x_centers, y_centers, h, colors = 'black', zorder = 2, norm = colors.LogNorm( vmin = 10.0**( int( np.log10( h.max() ) ) - 4 ), vmax = 10.0**( int( np.log10( h.max() ) ) + 1) ), levels = ( 10.0**( int( np.log10( h.max() ) ) - 4 ), 10.0**( int( np.log10( h.max() ) ) - 3 ), 10.0**( int( np.log10( h.max() ) ) - 2 ), 10.0**( int( np.log10( h.max() ) ) - 1 ), 10.0**( int( np.log10( h.max() ) ) + 0 ), 10.0**( int( np.log10( h.max() ) ) + 1 ) ),  linestyles = '--', linewidths = 1.0 )
    cs = ax.contourf(x_centers, y_centers, h, cmap = my_cmap,   zorder = 1, norm = colors.LogNorm( vmin = 10.0**( int( np.log10( h.max() ) ) - 4 ), vmax = 10.0**( int( np.log10( h.max() ) ) + 1) ), levels = ( 10.0**( int( np.log10( h.max() ) ) - 4 ), 10.0**( int( np.log10( h.max() ) ) - 3 ), 10.0**( int( np.log10( h.max() ) ) - 2 ), 10.0**( int( np.log10( h.max() ) ) - 1 ), 10.0**( int( np.log10( h.max() ) ) + 0 ), 10.0**( int( np.log10( h.max() ) ) + 1 ) ) )
    cbar = plt.colorbar(cs, ax=ax, shrink = 0.95, pad = 0.025, format = LogFormatter(10, labelOnlyBase=False))
    cs.set_clim( 10.0**( int( np.log10( h.max() ) ) - 4 ), 10.0**( int( np.log10( h.max() ) ) + 1 ) )
    # Ideal prediction
    ax.plot([data_min, data_max],[data_min, data_max], '-g', linewidth = 3, label='Ideal Prediction')
    plt.legend()
    # Configure plot
    # ax.set_xlim(xlim); #ax.set_xticks(xticks); 
    ax.set_xlabel('ground-truth')
    ax.tick_params( axis = 'x', direction = 'in', bottom = True, top = True, left = True, right = True )
    # ax.set_ylim(ylim); #plt.yticks(yticks); 
    ax.set_ylabel("prediction")
    ax.tick_params( axis = 'y', direction = 'in', bottom = True, top = True, left = True, right = True )
    ax.tick_params(axis = 'both', pad = 5) 	# add padding to both x and y axes, dist between axis ticks and label
    fig.savefig(f"figures/joinpdf_{target_name}_gt_vs_pred.png")
    plt.close(fig)