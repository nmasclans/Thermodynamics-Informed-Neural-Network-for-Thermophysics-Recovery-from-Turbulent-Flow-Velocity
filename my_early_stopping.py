import logging
import sys

import numpy as np


class my_EarlyStopping():
    """Stop training when a monitored metric has stopped improving.
    Assuming the goal of a training is to minimize the loss. With this, the
    metric to be monitored would be `'loss'`, and mode would be `'min'`. A
    `model.fit()` training loop will check at end of every epoch whether
    the loss is no longer decreasing, considering the `min_delta` and
    `patience` if applicable. Once it's found no longer decreasing,
    `stop_training` is marked True and the training should terminate.
    The quantity to be monitored needs to be available in `logs` dict.
    To make it so, pass the loss or metrics at `model.compile()`.
    Args:
      min_delta: Minimum change in the monitored quantity
          to qualify as an improvement, i.e. an absolute
          change of less than min_delta, will count as no
          improvement.
      patience: Number of epochs with no improvement
          after which training will be stopped.
      mode: One of `{"min", "max"}`. In `min` mode,
          training will stop when the quantity
          monitored has stopped decreasing; in `"max"`
          mode it will stop when the quantity
          monitored has stopped increasing.
      baseline: Baseline value for the monitored quantity.
          Training will stop if the model doesn't show improvement over the
          baseline.
      start_from_epoch: Number of epochs to wait before starting
          to monitor improvement. This allows for a warm-up period in which
          no improvement is expected and thus training will not be stopped.
    """

    def __init__(
        self,
        min_delta=0,
        patience=0,
        mode="min",
        baseline=None,
        start_from_epoch=0,
    ):
        self.patience = patience
        self.baseline = baseline
        self.min_delta = abs(min_delta)
        self.wait = 0
        self.stopped_epoch = 0
        self.best_epoch = 0
        self.start_from_epoch = start_from_epoch

        if mode not in ["auto", "min", "max"]:
            logging.warning(
                "EarlyStopping mode %s is unknown, fallback to auto mode.",
                mode,
            )
            mode = "auto"

        if mode == "min":
            self.monitor_op = np.less
        elif mode == "max":
            self.monitor_op = np.greater
        else:
            raise ValueError(f"Incorrect input value mode = {mode} 'my_EarlyStopping'. Accepted values: 'min', 'max'")

        if self.monitor_op == np.greater:
            self.min_delta *= 1
        else:
            self.min_delta *= -1

    def on_epoch_end(self, epoch, monitored_value):
        current = monitored_value
        self.wait += 1
        if self._is_improvement(current, self.best):
            self.best = current
            self.best_epoch = epoch
            if self.restore_best_weights:
                self.best_weights = self.model.get_weights()
            # Only restart wait if we beat both the baseline and our previous
            # best.
            if self.baseline is None or self._is_improvement(
                current, self.baseline
            ):
                self.wait = 0
        # Only check after the first epoch.
        if self.wait >= self.patience and epoch > 0:
            self.stopped_epoch = epoch
            sys.exit(f"Epoch {self.stopped_epoch + 1}: early stopping")
        
    def _is_improvement(self, monitor_value, reference_value):
        return self.monitor_op(monitor_value - self.min_delta, reference_value)