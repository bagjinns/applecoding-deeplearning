# import os
# print(len(os.listdir('../dataset/')))

import tensorflow as tf
import matplotlib.pyplot as plt

import os
from PIL import Image

root_dir = '/home/jin/applecoding-deeplearning/Image & CNN/kaggle_cat_VS_dog/PetImages/dataset'

for class_dir in os.listdir(root_dir):
    class_path = os.path.join(root_dir, class_dir)
    if not os.path.isdir(class_path):
        continue
    for fname in os.listdir(class_path):
        fpath = os.path.join(class_path, fname)
        try:
            img = Image.open(fpath)
            img.verify()  # 이미지 파일이 정상인지 검사
        except Exception as e:
            print(f"삭제됨: {fpath}  |  사유: {e}")
            os.remove(fpath)



train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    '/home/jin/applecoding-deeplearning/Image & CNN/kaggle_cat_VS_dog/PetImages/dataset',
    image_size=(64,64),
    batch_size=32,
    subset='training',
    validation_split=0.2,
    seed=1234
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    '/home/jin/applecoding-deeplearning/Image & CNN/kaggle_cat_VS_dog/PetImages/dataset',
    image_size=(64,64),
    batch_size=32,
    subset='validation',
    validation_split=0.2,
    seed=1234
)

print(train_ds)



# for i, j in train_ds.take(1):
#     print(i)
#     print(j)
#     plt.imshow(i[0].numpy().astype('uint8'))
#     plt.show()

model = tf.keras.Sequential([

    tf.keras.layers.RandomFlip('horizontal', input_shape=(64,64,3)),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.1),

    tf.keras.layers.Conv2D(32, (3,3), padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D((2,2)),
    tf.keras.layers.Conv2D(64, (3,3), padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D((2,2)),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Conv2D(128, (3,3), padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D((2,2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(1, activation='sigmoid'),
])


model.summary()

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(train_ds, validation_data=val_ds, epochs=5)