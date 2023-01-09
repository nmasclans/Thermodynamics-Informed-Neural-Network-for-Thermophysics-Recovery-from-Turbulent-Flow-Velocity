import sys

import tensorflow as tf

from tensorflow.keras import models
from my_visualizers import visualize_prediction, visualize_prediction_by_xyplanes
from my_utils import tf_print_time

# ----- Model Class: Multi-Layer Perceptron -----
class MLP(models.Model):
    def __init__(self, model, optimizer=None, loss_func=None, metric_func=[], epochs=100,
                 early_stopping=None, lr_scheduler=None, args={}, **kwargs):
        super(MLP, self).__init__(**kwargs)
        self.model = model # neural network 
        self.optimizer = optimizer # Adam optimizer
        self.epochs = epochs # number of epochs for training using Adam
        self.hist = []
        self.hist_test = []
        self.epoch = 0
        self.loss_func = loss_func
        self.metric_func = metric_func
        self.early_stopping = early_stopping
        self.lr_scheduler = lr_scheduler
        self.save_ckpt_freq = args.save_ckpt_freq
        if self.lr_scheduler is not None:
            self.lr_scheduler_call_frequency = self.lr_scheduler.get_call_frequency()
        else: 
            self.lr_scheduler_call_frequency = None
    
    @tf.function # (input_signature=(tf.TensorSpec(shape=(args.batch_size,args.num_features), dtype=tf.float32),
                 #                   tf.TensorSpec(shape=(args.batch_size,args.num_targets), dtype=tf.float32)))
    def train_step(self, x, y_gt):
        metric = []
        with tf.GradientTape() as tape:
            y_pred = self.model(x) 
            loss   = self.loss_func(y_gt, y_pred)
        trainable_vars = self.trainable_variables
        grads = tape.gradient(loss, trainable_vars)
        # for f in self.metric_func:
        #     metric.append(f(y_gt, y_pred))
        return grads, loss #, metric

    @tf.function # (input_signature=(tf.TensorSpec(shape=(args.batch_size_validation,args.num_features), dtype=tf.float32),
                 #                   tf.TensorSpec(shape=(args.batch_size_validation,args.num_targets), dtype=tf.float32)))
    def test_step(self, x, y_gt):
        metric = []
        y_pred = self.model(x)
        loss   = self.loss_func(y_gt, y_pred)
        for f in self.metric_func:
            metric.append(f(y_gt, y_pred))
        return y_pred, loss, metric

    def predict(self, dataset_pred, args):
        tf.print("\n-----------------------------------------------------------------------------")
        tf.print("Prediction")
        loss_val   = tf.Variable(0.0, dtype="float32")
        metric_val = tf.Variable(tf.zeros(args.num_metrics), dtype="float32")
        for nbatch, (features, targets_gt) in enumerate(dataset_pred):
            targets_pred, loss_batch, metric_batch = self.test_step(features, targets_gt)
            if args.make_plots:
                visualize_prediction(targets_gt, targets_pred, 0, nbatch, args)
                visualize_prediction_by_xyplanes(targets_gt, targets_pred, 0, nbatch, args)
            loss_val.assign_add(loss_batch) 
            metric_val.assign_add(metric_batch)
        loss_val.assign(loss_val/(nbatch+1))
        metric_val.assign(metric_val/(nbatch+1))
        tf.print(f"Loss '{args.loss}':",loss_val)
        tf.print(f"Metric {args.metrics}:\n",metric_val, summarize=-1)
        tf_print_time()

    def train_and_validate(self, dataset_tr, dataset_val, args):
        # --> training using Adam optimizer, validate at each epoch and visualize validation results
        for epoch in range(self.epochs):
            self.epoch = epoch
            # ---- train ----
            tf.print("\n-----------------------------------------------------------------------------")
            tf.print("Training Epoch:",epoch)
            loss_epoch = tf.Variable(0.0, dtype="float32")
            # Learning rate scheduler, if required
            if self.lr_scheduler is not None and self.lr_scheduler_call_frequency == 'epoch':
                self.lr_scheduler.on_epoch_begin(epoch)
            # Set Supervised_PINNS loss weights, if required
            if self.loss_func.name == "Supervised_PINNS":
                if self.epoch == 0:
                    self.loss_func.set_loss_weights(args.Supervised_PINNS_weights_first_epoch)
                elif self.epoch == 1:
                    self.loss_func.set_loss_weights(args.Supervised_PINNS_weights)
                else:
                    pass
            for nbatch, (features, targets_gt) in enumerate(dataset_tr):
                # Learning rate scheduler, if required
                if self.lr_scheduler is not None and self.lr_scheduler_call_frequency == 'batch':
                    self.lr_scheduler.on_batch_begin(epoch, nbatch)
                grads, loss_batch = self.train_step(features, targets_gt)
                self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
                loss_epoch.assign_add(loss_batch)
                # Print batch results, if required
                if nbatch % args.num_batches_per_print_information == 0 and nbatch != 0:
                    tf.print("    Batch:", nbatch, f"|| Loss '{args.loss}':", loss_batch) #, f"|| Metric {args.metrics}:", metric_batch)
                if tf.math.is_nan(loss_batch):
                    sys.exit(f"Loss takes NaN value at Epoch {epoch}, Batch {nbatch}. Consider using a smaller learning rate.")
            loss_epoch.assign(loss_epoch/(nbatch+1))
            self.hist.append(loss_epoch)
            tf.print(f"Loss '{args.loss}':",loss_epoch)
            tf_print_time()
            # Check early stopping, if required
            if self.early_stopping is not None:
                self.early_stopping.on_epoch_end(epoch, loss_epoch)
            
            # ---- validate ----
            tf.print("\n-----------------------------------------------------------------------------")
            tf.print("Validation Epoch",epoch)
            loss_val   = tf.Variable(0.0, dtype="float32")
            metric_val = tf.Variable(tf.zeros(args.num_metrics), dtype="float32")
            for nbatch, (features, targets_gt) in enumerate(dataset_val):
                targets_pred, loss_batch, metric_batch = self.test_step(features, targets_gt)
                if args.make_plots:
                    visualize_prediction(targets_gt, targets_pred, epoch, nbatch, args)
                loss_val.assign_add(loss_batch) 
                metric_val.assign_add(metric_batch)
            loss_val.assign(loss_val/(nbatch+1))
            metric_val.assign(metric_val/(nbatch+1))
            tf.print(f"Loss '{args.loss}':",loss_val)
            tf.print(f"Metric {args.metrics}:\n",metric_val, summarize=-1)
            tf_print_time()

            # ---- save weights at checkpoints ----
            if self.save_ckpt_freq != 0 and epoch % self.save_ckpt_freq == 0:
                 self.model.save_weights(f"checkpoints/ckpt_{epoch}")
