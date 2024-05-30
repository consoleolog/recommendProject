import pandas as pd
import numpy as np
import tensorflow as tf
import shutil
import os
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
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

#----------------------------------------  이미지 전처리 ----------------------------------------

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

# img_generator = ImageDataGenerator(
#     rescale=1./255,
#     rotation_range=20, # 회전
#     zoom_range=0.15, # 확대
#     width_shift_range=0.2, # 이동
#     height_shift_range=0.2,
#     shear_range=0.15,  # 굴정
#     horizontal_flip=True, # 가로 반전
#     fill_mode="nearest"
# )
#
# train_generator = img_generator.flow_from_directory(
#     'dataset',
#     class_mode='binary',  # 두개면 binary, 몇 개 더면 categorical
#     shuffle=True,
#     seed=123,
#     color_mode='rgb',
#     batch_size=64,
#     target_size=(128, 128)
# )




# print(train_ds) # <_PrefetchDataset element_spec=(TensorSpec(shape=(None, 128, 128, 3), dtype=tf.float32, name=None), TensorSpec(shape=(None,), dtype=tf.int32, name=None))>

def normalize(i, answer):
    i = tf.cast(i / 255.0, tf.float32)
    return i, answer


# 노멀라이징 필수임
train_ds = train_ds.map(normalize)
val_ds = val_ds.map(normalize)









#--------------------------------- 숫자 데이터 전처리 -----------------------------

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








#------------------------- 해시태그 전처리 -------------------------------

hashtag_string_lookup_layer = tf.keras.layers.StringLookup(
    vocabulary=result['hashtag'].unique(),
    num_oov_indices=0,
    output_mode='one_hot'
)
hashtag_one_hot = hashtag_string_lookup_layer(result['hashtag'])









#---------------------------- 사용자 이름 전처리 ------------------------------

username_string_lookup_layer = tf.keras.layers.StringLookup(
    vocabulary=result['username'].unique(),
    num_oov_indices=0,
    output_mode='one_hot'
)

embedding = tf.keras.layers.Embedding(len(result['username'].unique()), 4)
username_indices = username_string_lookup_layer(result['username'])
username_embedded = embedding(username_indices)








# ---------------------- 이미지 모델 --------------------------

Image_Input = tf.keras.Input(shape=(128, 128, 3))
Image_Conv2D_1 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu')(Image_Input)
Image_MaxPool2D_1 = tf.keras.layers.MaxPooling2D((2, 2))(Image_Conv2D_1)
Image_Dropout_1 = tf.keras.layers.Dropout(0.2)(Image_MaxPool2D_1)
Image_Conv2D_2 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu')(Image_Dropout_1)
Image_MaxPool2D_2 = tf.keras.layers.MaxPooling2D((2, 2))(Image_Conv2D_2)
Image_Flatten = tf.keras.layers.Flatten()(Image_Conv2D_2)
Image_Dense_1 = tf.keras.layers.Dense(64, activation='relu')(Image_Flatten)
Image_Dropout_2 = tf.keras.layers.Dropout(0.2)(Image_Dense_1)
Image_Output = tf.keras.layers.Dense(32, activation='relu')(Image_Dropout_2)

Image_Model = tf.keras.models.Model(Image_Input, Image_Output)



# ---------------------------- 해시태그 모델 -------------------------------

Hashtag_Input = tf.keras.Input(shape=(2, 1))
Hashtag_Dense_1 = tf.keras.layers.Dense(64, activation='relu')(Hashtag_Input)
Hashtag_Dense_2 = tf.keras.layers.Dense(32, activation='relu')(Hashtag_Dense_1)
Hashtag_Flatten = tf.keras.layers.Flatten()(Hashtag_Dense_2)
Hashtag_Output = tf.keras.layers.Dense(32, activation='relu')(Hashtag_Flatten)

Hashtag_Model = tf.keras.models.Model(Hashtag_Input, Hashtag_Output)


# ------------------------- 숫자 데이터 모델 ---------------------------
Number_Input = tf.keras.Input(shape=(2, 3))
Number_Dense_1 = tf.keras.layers.Dense(128, activation='relu')(Number_Input)
Number_Flatten = tf.keras.layers.Flatten()(Number_Dense_1)
Number_Output = tf.keras.layers.Dense(32, activation='relu')(Number_Flatten)

Number_Model = tf.keras.models.Model(Number_Input, Number_Output)



#------------------------------- 사용자 이름 모델 ---------------------------------
Username_Input = tf.keras.Input(shape=(2, 2))
Username_Flatten = tf.keras.layers.Flatten()(Username_Input)
Username_Output = tf.keras.layers.Dense(32, activation='relu')(Username_Flatten)

Username_Model = tf.keras.models.Model(Username_Input, Username_Output)


Combined = tf.keras.layers.concatenate([Hashtag_Model.output, Number_Model.output, Username_Model.output, Image_Model.output])
x = tf.keras.layers.Dense(32, activation='relu')(Combined)
x = tf.keras.layers.Dense(16, activation='relu')(x)
Output = tf.keras.layers.Dense(1, activation='sigmoid')(x)

model = tf.keras.models.Model([Hashtag_Input, Number_Input, Username_Input, Image_Input], Output)

model.summary()

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

#tf.keras.utils.plot_model(model, to_file='model.png',show_shapes=True, show_layer_names=True)

label = result['label'].values

def generator(image_dataset, text_dataset, answer):
    for images, labels in image_dataset:
        text_batch = text_dataset[:len(images)]
        text_labels_batch = answer[:len(images)]
        yield ([images, text_batch], text_labels_batch)

combined_dataset = tf.data.Dataset.from_generator(
    generator,
    output_signature=(
        (tf.TensorSpec(shape=(None,128,128,3), dtye=tf.float32)
         ),

    ),
    args=()
)






exit()
x = [train_ds, username_embedded, input_data, hashtag_one_hot]

model.fit(x, batch_size=32, epochs=10)














