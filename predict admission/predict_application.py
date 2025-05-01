import tensorflow as tf
import pandas as pd
import numpy as np

df = pd.read_csv('gpascore.csv')

# print(df.isnull().sum())
df = df.dropna()
# print(df.isnull().sum())


# for i, rows in df.iterrows():
#     x.append([rows['gre'], rows['gpa'], rows['rank']])
# exit()

x = df.drop('admit', axis=1)
y = df['admit'].values
  



model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(64, activation='tanh'),
    tf.keras.layers.Dense(128, activation='tanh'),
    tf.keras.layers.Dense(1, activation='sigmoid'), #sigmoid activation function is working for make the results between 0 to 1
])


model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

model.fit(np.array(x), np.array(y), epochs=1000)


#predict
predicted = model.predict(np.array([[750, 3.7, 3], [400, 2.2, 1]]))
print(predicted)