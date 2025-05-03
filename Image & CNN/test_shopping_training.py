import tensorflow as tf
import numpy as np




(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

x_train = x_train / 255.0
x_test = x_test / 255.0

x_train = x_train.reshape((x_train.shape[0], 28, 28, 1))
x_test = x_test.reshape((x_test.shape[0], 28, 28, 1))


called_model = tf.keras.models.load_model('/home/jin/applecoding-deeplearning/Image & CNN/kaggle_cat_VS_dog/model1.keras')
called_model.summary()

called_model.evaluate(x_test, y_test)






exit()

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

x_train = x_train / 255.0
x_test = x_test / 255.0

x_train = x_train.reshape((x_train.shape[0], 28, 28, 1))
x_test = x_test.reshape((x_test.shape[0], 28, 28, 1))

model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28, 1)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax'),
])

model.summary()

# model.compile(loss='sparese_catergorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=3)

model.save('/home/jin/applecoding-deeplearning/Image & CNN/kaggle_cat_VS_dog/model1.keras')
