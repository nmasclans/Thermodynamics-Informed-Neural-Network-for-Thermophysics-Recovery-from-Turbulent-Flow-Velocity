import tensorflow as tf
import numpy as np

from tensorflow.keras import backend

class my_LearningRateScheduler():
    """Learning rate scheduler.
    At the beginning of every epoch, this callback gets the updated learning
    rate value from `schedule` function provided at `__init__`, with the current
    epoch and current learning rate, and applies the updated learning rate on
    the optimizer.
    Args:
      schedule_function: a function that takes an epoch index (integer, indexed from 0)
          and current learning rate (float) as inputs and returns a new
          learning rate as output (float).
      verbose: int. 0: quiet, 1: update messages.
    """

    def __init__(self, scheduler_function, optimizer, call_frequency='epoch',
                 num_batches_per_epoch = None, verbose=0):
        self.scheduler = scheduler_function
        self.optimizer = optimizer
        self.verbose = verbose
        assert call_frequency in ['epoch', 'batch'], \
            f"Invalid my_LearningRateScheduler input 'call_frequency' = {call_frequency}. "\
            "Accepted values: 'epoch', 'batch'"
        self.call_frequency = call_frequency
        if call_frequency == 'batch':
            assert num_batches_per_epoch is not None, \
            f"Invalid my_LearningRateScheduler input 'num_batches_per_epoch' = '{num_batches_per_epoch}'. "\
            "For input 'call_frequency' = 'batch', 'num_batches_per_epoch' must be Integer"
            self.num_batches_per_epoch = num_batches_per_epoch


    def on_epoch_begin(self, epoch):
        lr = self.scheduler(epoch)
        if not isinstance(lr, (tf.Tensor, float, np.float32, np.float64)):
            raise ValueError(
                'The output of the "schedule" function '
                f"should be float. Got: {lr}"
            )
        if isinstance(lr, tf.Tensor) and not lr.dtype.is_floating:
            raise ValueError(
                f"The dtype of `lr` Tensor should be float. Got: {lr.dtype}"
            )
        backend.set_value(self.optimizer.lr, backend.get_value(lr))
        if self.verbose > 0:
            print(
                f"\nEpoch {epoch}: LearningRateScheduler setting learning "
                f"rate to {lr}."
            )

    def on_batch_begin(self, epoch, batch):
        num_batch_total = self.num_batches_per_epoch*epoch + batch
        lr = self.scheduler(num_batch_total)
        if not isinstance(lr, (tf.Tensor, float, np.float32, np.float64)):
            raise ValueError(
                'The output of the "schedule" function '
                f"should be float. Got: {lr}"
            )
        if isinstance(lr, tf.Tensor) and not lr.dtype.is_floating:
            raise ValueError(
                f"The dtype of `lr` Tensor should be float. Got: {lr.dtype}"
            )
        backend.set_value(self.optimizer.lr, backend.get_value(lr))
        if self.verbose > 0:
            print(
                f"\nEpoch {epoch} / Batch {batch}: LearningRateScheduler setting learning "
                f"rate to {lr:.3e}."
            )

    def get_call_frequency(self):
        return self.call_frequency