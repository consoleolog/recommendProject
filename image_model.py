import pandas as pd
import numpy as np
import tensorflow as tf
import shutil
import os
import matplotlib.pyplot as plt

post_data = pd.read_csv('./data/post_data.csv')
user_data = pd.read_csv('./data/user_data.csv')
image_data = pd.read_csv('./data/image_data.csv')
post = pd.merge(left=post_data, right=image_data, how='inner', on=['pk', 'post_id'])
result = pd.merge(left=post, right=user_data, how='inner', on=['username'])


def calculate_label(row):
    likes_condition = row['like_count'] / row['follower_count'] >= 1
    comments_condition = row['like_count'] / row['comment_count'] >= 1
    followers_condition = row['follower_count'] / row['following_count'] >= 1
    label = int(likes_condition and comments_condition and followers_condition)
    return label


result['label'] = result.apply(calculate_label, axis=1)

# 반복문이 이미지 all 을 돌아야되는거임 돌아가면서 이름이 나오겠지?
for filename in os.listdir('./image_all'):
    row = result[result['image_name'] == filename]
    if not row.empty:
        label = row['label'].values[0]
        if label == 0:
            # 이 파일 네임과 일치하는 데이터 프레임의 한 행을 찾아야돼
            # 그리고 그 데이터 프레임의 라벨을 확인하고 0이면
            shutil.copyfile('./image_all/' + filename, './dataset/image_0/' + filename)
        elif label == 1:
            # 데이터 프레임의 라벨이 1이면
            shutil.copyfile('./image_all/' + filename, './dataset/image_1/' + filename)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    'dataset',
    image_size=(128, 128),
    batch_size=32,
    subset='validation',
    seed=123,
    validation_split=0.2,
)
# 이미지 전처리
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    'dataset',
    image_size=(128, 128),
    batch_size=32,
    subset='training',
    seed=123,
    validation_split=0.2,
)

# print(train_ds) # <_PrefetchDataset element_spec=(TensorSpec(shape=(None, 128, 128, 3), dtype=tf.float32, name=None), TensorSpec(shape=(None,), dtype=tf.int32, name=None))>

def normalize(i, answer):
    i = tf.cast(i / 255.0, tf.float32)
    return i, answer


# 노멀라이징 필수임
train_ds = train_ds.map(normalize)
val_ds = val_ds.map(normalize)

Input = tf.keras.Input(shape=(128, 128, 3))
Conv2D_1 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu')(Input)
MaxPool2D_1 = tf.keras.layers.MaxPooling2D((2, 2))(Conv2D_1)
Dropout_1 = tf.keras.layers.Dropout(0.2)(MaxPool2D_1)
Conv2D_2 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu')(Dropout_1)
MaxPool2D_2 = tf.keras.layers.MaxPooling2D((2, 2))(Conv2D_2)
Flatten = tf.keras.layers.Flatten()(Conv2D_2)
Dense_1 = tf.keras.layers.Dense(64, activation='relu')(Flatten)
Dropout_2 = tf.keras.layers.Dropout(0.2)(Dense_1)
Output = tf.keras.layers.Dense(1, activation='sigmoid')(Dropout_2)

es = tf.keras.callbacks.EarlyStopping(
    monitor='accuracy',
    patience=5,
    mode='max',
)

model = tf.keras.models.Model(Input, Output)

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(train_ds, validation_data=val_ds, epochs=100, callbacks=[es])

model.save('./models/image_model')






