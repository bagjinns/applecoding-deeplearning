# import os
# print(len(os.listdir('../dataset/')))

import tensorflow as tf
import matplotlib.pyplot as plt


train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    '/Users/jinseopark/Desktop/python/applecoding-deeplearning/Image & CNN/kaggle_cat_VS_dog/PetImages/dataset',
    image_size=(64,64),
    batch_size=32,
    subset='training',
    validation_split=0.2,
    seed=1234
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    '/Users/jinseopark/Desktop/python/applecoding-deeplearning/Image & CNN/kaggle_cat_VS_dog/PetImages/dataset',
    image_size=(64,64),
    batch_size=32,
    subset='validation',
    validation_split=0.2,
    seed=1234
)

print(train_ds)



for i, j in train_ds.take(1):
    print(i)
    print(j)
    plt.imshow(i[0].numpy().astype('uint8'))
    plt.show()