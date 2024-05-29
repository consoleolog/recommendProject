import pandas as pd
import numpy as np
import tensorflow as tf
import shutil
import os

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
    print(filename)
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


def normalize(i, answer):
    i = tf.cast(i / 255.0, tf.float32)
    return i, answer


train_ds = train_ds.map(normalize)

val_ds = val_ds.map(normalize)

input_data = []
for i, rows in result.iterrows():
    input_data.append(
        [
            rows['follower_count'],
            rows['following_count'],
            rows['media_count']
        ]
    )
normalization_layer = tf.keras.layers.Normalization(mean=2.0, variance=1.0)
input_data = normalization_layer(tf.constant(input_data))

hashtag_string_lookup_layer = tf.keras.layers.StringLookup(
    vocabulary=result['hashtag'].unique(),
    num_oov_indices=0,
    output_mode='one_hot'
)
hashtag_one_hot = hashtag_string_lookup_layer(result['hashtag'])

username_string_lookup_layer = tf.keras.layers.StringLookup(
    vocabulary=result['username'].unique(),
    num_oov_indices=0,
    output_mode='one_hot'
)

embedding = tf.keras.layers.Embedding(len(result['username'].unique()), 4)
username_indices = username_string_lookup_layer(result['username'])
username_embedded = embedding(username_indices)


print(type(username_embedded))
print(type(input_data))
print(type(hashtag_one_hot))


dataset1 = tf.data.Dataset.from_tensor_slices(username_embedded)
dataset2 = tf.data.Dataset.from_tensor_slices(input_data)
dataset3 = tf.data.Dataset.from_tensor_slices(hashtag_one_hot)
map_dataset1 = dataset3.map(lambda x:x+1)
map_dataset2 = dataset3.map(lambda x:x+1)
map_dataset3 = dataset3.map(lambda x:x+1)
print(type(map_dataset1))
print(type(map_dataset2))
print(type(map_dataset3))

combined_dataset = tf.data.Dataset.zip((dataset1, dataset2, dataset3))
print(type(combined_dataset))


Username_Input = tf.keras.Input(shape=(2, 2))
Username_Output = tf.keras.layers.Dense(32, activation='relu')(Username_Input)

Number_Input = tf.keras.Input(shape=(2, 3))
Number_Dense_1 = tf.keras.layers.Dense(128, activation='relu')(Number_Input)
Number_Output = tf.keras.layers.Dense(32, activation='relu')(Number_Dense_1)

Hashtag_Input = tf.keras.Input(shape=(2, 1))
Hashtag_Dense_1 = tf.keras.layers.Dense(64, activation='relu')(Hashtag_Input)
Hashtag_Dense_2 = tf.keras.layers.Dense(32, activation='relu')(Hashtag_Dense_1)
Hashtag_Output = tf.keras.layers.Dense(32, activation='relu')(Hashtag_Dense_2)

Image_Input = tf.keras.Input(shape=(128, 128, 3), name='image_input')
Con2d_1 = tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu')(Image_Input)
MaxPool2D_1 = tf.keras.layers.MaxPooling2D((2, 2))(Con2d_1)
global_avg_pooling = tf.keras.layers.GlobalAveragePooling2D()(MaxPool2D_1)
Flatten_1 = tf.keras.layers.Flatten()(MaxPool2D_1)
Image_Output = tf.keras.layers.Dense(32, activation='relu')(Flatten_1)

Combined_1 = tf.keras.layers.Concatenate(axis=-1)([Username_Output, Number_Output, Hashtag_Output])
Combined_1 = tf.keras.layers.Dense(32, activation='relu')(Combined_1)
Combined_1 = tf.keras.layers.Flatten()(Combined_1)
Combined_1 = tf.keras.layers.Dense(32, activation='relu')(Combined_1)

Combined_2 = tf.keras.layers.Concatenate(axis=-1)([Combined_1, Image_Output])
x = tf.keras.layers.Dense(32, activation='relu')(Combined_2)
Output = tf.keras.layers.Dense(1, activation='sigmoid')(x)

model = tf.keras.models.Model(inputs=[Username_Input, Number_Input, Hashtag_Input, Image_Input], outputs=Output)
model.summary()
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

X_data = [map_dataset1, map_dataset2, map_dataset3, train_ds]
Y_data = result['label'].values
Y_data = tf.data.Dataset.from_tensor_slices(Y_data)
map_Y_data = Y_data.map(lambda x:x+1)

es = tf.keras.callbacks.EarlyStopping(
    monitor='accuracy',
    patience=5,
    mode='max',
)

model.fit(x=X_data, y=map_Y_data, epochs=100, callbacks=[es], batch_size=32)

model.save('./saved_models/model1.h5')
