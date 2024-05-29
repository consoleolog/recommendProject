import pandas as pd
import numpy as np
import tensorflow as tf

result = pd.read_csv('./data/predict_data.csv')

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    'predict_image',
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
input_data = normalization_layer(input_data)

hashtag_string_lookup_layer = tf.keras.layers.StringLookup(
    vocabulary=result['hashtag'].unique,
    num_oov_indices=0,
    output_mode='one_hot'
)
hashtag_one_hot = hashtag_string_lookup_layer(result['hashtag'])

username_string_lookup_layer = tf.keras.layers.StringLookup(
    vocabulary=result['username'].unique,
    num_oov_indices=0,
    output_mode='one_hot'
)
embedding = tf.keras.layers.Embedding(len(result['username'].unique), 4)
username_indices = username_string_lookup_layer(result['username'])
username_embedding = embedding(username_indices)

Username_Input = tf.keras.Input(shape=(2, 2))
