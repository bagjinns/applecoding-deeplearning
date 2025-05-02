import tensorflow as tf
import pandas as pd
import numpy as np


# tensor = tf.constant([3, 4, 5])
# tensor2 = tf.constant([5, 6, 7])

# tensor3 = tf.constant([[1, 2, 3],
                    #    [3, 4, 5]])
# print(tensor + tensor2)
# tensor4 = tf.zeros([2,2, 3])
# print(tensor4.shape)

w = tf.Variable(1.0)
w.assign(2)
print(w.numpy())