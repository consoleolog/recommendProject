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

#print(train_ds) # <_PrefetchDataset element_spec=(TensorSpec(shape=(None, 128, 128, 3), dtype=tf.float32, name=None), TensorSpec(shape=(None,), dtype=tf.int32, name=None))>

# for i, answer in train_ds.take(1):
#     print(answer)
#     plt.imshow(i[0].numpy().astype('uint8'))
#     plt.show()

def normalize(i, answer):
    i = tf.cast(i / 255.0, tf.float32)
    return i, answer
# 노멀라이징 필수임
train_ds = train_ds.map(normalize)
print(train_ds) # <_MapDataset element_spec=(TensorSpec(shape=(None, 128, 128, 3), dtype=tf.float32, name=None), TensorSpec(shape=(None,), dtype=tf.int32, name=None))>

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
print(input_data) # shape=(2,3) 2 x 3 행렬

hashtag_string_lookup_layer = tf.keras.layers.StringLookup(
    vocabulary=result['hashtag'].unique(),
    num_oov_indices=0,
    output_mode='one_hot'
)
hashtag_one_hot = hashtag_string_lookup_layer(result['hashtag'])

print(hashtag_one_hot) #shape=(2,1) 2x1 행렬

username_string_lookup_layer = tf.keras.layers.StringLookup(
    vocabulary=result['username'].unique(),
    num_oov_indices=0,
    output_mode='one_hot'
)

embedding = tf.keras.layers.Embedding(len(result['username'].unique()), 4)
username_indices = username_string_lookup_layer(result['username'])
username_embedded = embedding(username_indices)

print(username_embedded) # shape=(2,2,4)

# 4차원 , 2차원, 2차원 , 3차원
Y_data = result['label'].values
Y_data = tf.data.Dataset.from_tensor_slices(Y_data)
username_embedded = tf.data.Dataset.from_tensor_slices(username_embedded)
input_data = tf.data.Dataset.from_tensor_slices(input_data)
hashtag_one_hot = tf.data.Dataset.from_tensor_slices(hashtag_one_hot)
# train_ds = tf.data.Dataset.from_tensor_slices(train_ds)


ds = tf.data.Dataset.zip(username_embedded,input_data,hashtag_one_hot,train_ds,Y_data)
print(ds)

Username_Input = tf.keras.Input(shape=(2, 2, 4))
Username_Dense = tf.keras.layers.Dense(16, activation='relu')(Username_Input)
Username_Flatten = tf.keras.layers.Flatten()(Username_Dense)
Username_Output = tf.keras.layers.Dense(32, activation='relu')(Username_Flatten)

Number_Input = tf.keras.Input(shape=(2, 3))
Number_Dense = tf.keras.layers.Dense(16, activation='relu')(Number_Input)
Number_Flatten = tf.keras.layers.Flatten()(Number_Dense)
Number_Output = tf.keras.layers.Dense(32, activation='relu')(Number_Flatten)

Hashtag_Input = tf.keras.Input(2,1)
Hashtag_Dense = tf.keras.layers.Dense(16, activation='relu')(Hashtag_Input)
Hastag_Flatten = tf.keras.layers.Flatten()(Hashtag_Dense)
Hashtag_Output = tf.keras.layers.Dense(32, activation='relu')(Hastag_Flatten)

Image_Input = tf.keras.Input(shape=(128, 128, 3), name='image_input')
Con2d_1 = tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu')(Image_Input)
MaxPool2D_1 = tf.keras.layers.MaxPooling2D((2, 2))(Con2d_1)
global_avg_pooling = tf.keras.layers.GlobalAveragePooling2D()(MaxPool2D_1)
Flatten_1 = tf.keras.layers.Flatten()(MaxPool2D_1)
Image_Output = tf.keras.layers.Dense(32, activation='relu')(Flatten_1)

# #2차원 먼저 합쳐
# Demension2 = tf.keras.layers.Concatenate(axis=-1)([Number_Output, Hashtag_Output])
# Concat_Dense = tf.keras.layers.Dense(32, activation='relu')(Demension2)
# Concat_Flatten = tf.keras.layers.Flatten()(Concat_Dense)
# Concat_Output = tf.keras.layers.Dense(32, activation='relu')(Concat_Flatten)

Combine = tf.keras.layers.Concatenate(axis=-1)([Number_Output,Image_Output,Hashtag_Dense,Username_Output])
x = tf.keras.layers.Dense(32, activation='relu')(Combine)
Output = tf.keras.layers.Dense(1, activation='sigmoid')(x)
#나머지
# Concat = tf.keras.layers.Concatenate([Concat_Output,Image_Output, Username_Output])
# x = tf.keras.layers.Dense(64, activation='relu')(Concat)
# x = tf.keras.layers.Dense(32, activation='relu')(x)
# Output = tf.keras.layers.Dense(1, activation='sigmoid')(x)
model = tf.keras.models.Model(inputs=[Image_Input,Hashtag_Input,Username_Input,Number_Input],outputs=Output)
model.summary()
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
Y_data = result['label'].values
ds_batch = ds.batch(32)
es = tf.keras.callbacks.EarlyStopping(
    monitor='accuracy',
    patience=5,
    mode='max',
)

model.fit(ds_batch, epochs=100, callbacks=[es])
exit()

es = tf.keras.callbacks.EarlyStopping(
    monitor='accuracy',
    patience=5,
    mode='max',
)

model.fit(ds_batch, epochs=100, callbacks=[es], batch_size=32)

model.save('./saved_models/model1.h5')


