import tensorflow as tf
from datetime import datetime

def tf_print_time():
    now = datetime.now().strftime("%H:%M:%S")
    tf.print("Current Time:",now)