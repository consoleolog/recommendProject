import pandas as pd
import numpy as np
import tensorflow as tf
import shutil
import os
from PIL import Image

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

image_0_list = os.listdir('./dataset/image_0')

image_1_list = os.listdir('./dataset/image_1')

image0_numList = []
for filename in image_0_list:
    convert_num = Image.open(f'./dataset/image_0/{filename}').convert('RGB').crop((20,30,160,180)).resize((64,64))
    image0_numList.append(np.array(convert_num))

image1_numList = []
for filename in image_1_list:
    convert_num = Image.open(f'./dataset/image_1/{filename}').convert('RGB').crop((20,30,160,180)).resize((64,64))
    image1_numList.append(np.array(convert_num))

image0_numList = np.divide(image0_numList, 255)
image1_numList = np.divide(image1_numList, 255)

