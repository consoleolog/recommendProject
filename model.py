import pandas as pd
import numpy as np
import tensorflow as tf
import shutil
import os

# 이미지 전처리
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    './image_data',
    image_size=(128, 128),
    batch_size=32,
    subset='training',
    seed=123,
    validation_split=0.2,
)
val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    './image_data',
    image_size=(128, 128),
    batch_size=32,
    subset='validation',
    seed=123,
    validation_split=0.2,
)
def normalize(i, answer):
    i = tf.cast(i / 255.0, tf.float32)
    return i, answer


train_ds = train_ds.map(normalize)
val_ds = val_ds.map(normalize)

Image_Input = tf.keras.layers.Input(shape=(128, 128, 3), name='image_input')
Con2d_1 = tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu')(Image_Input)
MaxPool2D_1 = tf.keras.layers.MaxPooling2D((2, 2))(Con2d_1)
Flatten_1 = tf.keras.layers.Flatten()(MaxPool2D_1)
Image_Output = tf.keras.layers.Dense(1, activation='sigmoid')(Flatten_1)


post_data = pd.read_csv('./data/post_data.csv')
user_data = pd.read_csv('./data/user_data.csv')
image_data = pd.read_csv('./data/image_data.csv')
post = pd.merge(left=post_data, right=image_data, how='inner', on=['pk', 'post_id'])
result = pd.merge(left=post, right=user_data, how='inner', on=['username'])

like_count = result['like_count']
follower_count = result['follower_count']
following_count = result['following_count']
comment_count = result['comment_count']

def calculate_label(row):
    likes_condition = row['likes'] / row['follower_count'] >= 1
    comments_condition = row['comment_count'] / row['comment_count'] >= 1
    followers_condition = row['follower_count'] / row['following_count'] >= 1

    label = int(likes_condition and comments_condition and followers_condition)
    return label


result['label'] = result.apply(calculate_label, axis=1)

# 반복문이 이미지 all 을 돌아야되는거임 돌아가면서 이름이 나오겠지?
for filename in os.listdir('./image_all'):
    print(filename)
    row = result[result['photos'] == filename]
    if not row.empty:
        label = row['label'].values[0]
        if label == 0:
            # 이 파일 네임과 일치하는 데이터 프레임의 한 행을 찾아야돼
            # 그리고 그 데이터 프레임의 라벨을 확인하고 0이면
            shutil.copyfile('./image_all/' + filename, './image_data/image_0/' + filename)
        elif label == 1:
            # 데이터 프레임의 라벨이 1이면
            shutil.copyfile('./image_all/' + filename, './image_data/image_1/' + filename)

input_data = []

for i, rows in result.iterrows():
    input_data.append(
        [
            rows['username'],
            rows['hashtag'],
            rows['post_date'],
            rows['follower_count'],
            rows['following_count'],
            rows['media_count']
        ]
    )

Y_data = result['label'].values

Input = tf.keras.layers.Input(shape=(3, 3))
Dense_1 = tf.keras.layers.Dense(128, activation='relu')(Input)
Dense_2 = tf.keras.layers.Dense(64, activation='relu')(Dense_1)
Dense_3 = tf.keras.layers.Dense(32, activation='relu')(Dense_2)
Output = tf.keras.layers.Dense(1, activation='sigmoid')(Dense_3)

Combined = tf.keras.layers.Concatenate([Image_Output, Output])
x = tf.keras.layers.Dense(64, activation='relu')(Combined)
x = tf.keras.layers.Dense(32, activation='relu')(x)
Real_Output = tf.keras.layers.Dense(1, activation='sigmoid')(x)

model = tf.keras.models.Model(inputs=[Image_Input, Input], outputs=Real_Output)

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

es = tf.keras.callbacks.EarlyStopping(
    monitor='accuracy',
    patience=5,
    mode='max',
)

model.fit([train_ds, input_data, Y_data], epochs=100, callbacks=[es], batch_size=32)

model.save('./saved_models/model1.h5')
