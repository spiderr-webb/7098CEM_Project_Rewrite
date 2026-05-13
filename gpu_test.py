import tensorflow as tf


print("Tensorflow Version: ", tf.version.VERSION)
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
