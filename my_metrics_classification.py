import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa


NUM_CLASSES = 3


def compute_metrics(y_gt, y_pred):
    compute_metric_IoU(y_gt, y_pred)
    compute_metric_accuracy(y_gt, y_pred)

def compute_metric_IoU(y_gt, y_pred):
    per_class_iou = []
    for c in range(NUM_CLASSES):
        m_i = tf.keras.metrics.IoU(num_classes = NUM_CLASSES, target_class_ids = [c])
        m_i.update_state(y_gt, y_pred)
        per_class_iou.append(m_i.result().numpy())
    m = tf.keras.metrics.MeanIoU(num_classes = NUM_CLASSES)
    m.update_state(y_gt, y_pred)
    mean_iou = m.result().numpy()
    tf.print(f"\nWeighted IoU: {mean_iou:.3f}")
    tf.print(f"Per-Class IoU: \n    liquid-like: {per_class_iou[0]:.3f} \n    two-phases-like: {per_class_iou[1]:.3f} \n    gas-like: {per_class_iou[2]:.3f}\n")


def compute_metric_accuracy(y_gt, y_pred):
    m = tf.keras.metrics.Accuracy()
    m.update_state(y_gt, y_pred)
    tf.print(f"Accuracy: {m.result().numpy():.3f}\n")

"""
def compute_metric_F1Score(y_gt, y_pred):
    y_gt   = y_gt.astype(np.int32)
    y_pred = y_pred.astype(np.float32)
    tf.print(y_gt.shape)
    tf.print(y_pred.shape)
    mean_F1      = tfa.metrics.F1Score(num_classes = NUM_CLASSES, average='weighted')
    per_class_F1 = tfa.metrics.F1Score(num_classes = NUM_CLASSES, average=None)
    tf.print(y_pred)
    mean_F1.update_state(y_gt, y_pred)
    per_class_F1.update_state(y_gt, y_pred)
    tf.print(f"\nWeigthed F1 Score: {mean_F1.result().numpy():.3f}")
    tf.print(f"Per-Class F1 Score: \n    liquid-like: {per_class_F1[0]:.3f} \n    two-phases-like: {per_class_F1[1]:.3f} \n    gas-like: {per_class_F1[2]:.3f}\n")
"""