import tensorflow as tf
import numpy as np

from keras import backend

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

    def __init__(self, schedule_function, optimizer, verbose=0):
        self.schedule = schedule_function
        self.optimizer = optimizer
        self.verbose = verbose

    def on_epoch_begin(self, epoch):
        try:  # new API
            lr = float(backend.get_value(self.optimizer.lr))
            lr = self.schedule(epoch, lr)
        except TypeError:  # Support for old API for backward compatibility
            lr = self.schedule(epoch)
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
                f"\nEpoch {epoch + 1}: LearningRateScheduler setting learning "
                f"rate to {lr}."
            )
