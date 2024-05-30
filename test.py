import tensorflow as tf

# Step 1: Create an initial dataset and a MapDataset
initial_dataset = tf.data.Dataset.from_tensor_slices([1, 2, 3, 4, 5])

print(type(initial_dataset))
def map_fn(x):
    return x * 2  # Example transformation

map_dataset = initial_dataset.map(map_fn)

print(type(map_dataset))

# Step 2: Iterate through the MapDataset and collect the elements
elements = []

for element in map_dataset:
    elements.append(element)

eager_tensor = tf.stack(elements)

# Print the resulting EagerTensor
print(type(eager_tensor))  # Should output something like tf.Tensor([ 2  4  6  8 10], shape=(5,), dtype=int32)



exit()
# 데이터 준비
data_4d = tf.constant([[[[1, 2], [3, 4]], [[5, 6], [7, 8]]],
                      [[[9, 10], [11, 12]], [[13, 14], [15, 16]]]])
data_2d = tf.constant([[1, 2, 3],
                      [4, 5, 6]])
data_3d = tf.constant([[[1, 2, 3], [4, 5, 6]],
                      [[7, 8, 9], [10, 11, 12]]])
labels = tf.constant([0, 1])

# 데이터셋 생성
dataset_4d = tf.data.Dataset.from_tensor_slices(data_4d)
dataset_2d = tf.data.Dataset.from_tensor_slices(data_2d)
dataset_3d = tf.data.Dataset.from_tensor_slices(data_3d)
label_dataset = tf.data.Dataset.from_tensor_slices(labels)

# 데이터셋을 묶음
combined_dataset = tf.data.Dataset.zip((dataset_4d, dataset_2d, dataset_3d))

# 모델 생성
inputs_4d = tf.keras.layers.Input(shape=(2, 2, 2))
inputs_2d = tf.keras.layers.Input(shape=(3,))
inputs_3d = tf.keras.layers.Input(shape=(2, 3))
flatten_4d = tf.keras.layers.Flatten()(inputs_4d)
flatten_3d = tf.keras.layers.Flatten()(inputs_3d)
concatenated_inputs = tf.keras.layers.Concatenate()([flatten_4d, inputs_2d, flatten_3d])
outputs = tf.keras.layers.Dense(1)(concatenated_inputs)
model = tf.keras.Model(inputs=[inputs_4d, inputs_2d, inputs_3d], outputs=outputs)

# 모델 컴파일
model.compile(optimizer='adam', loss='mse')

# 입력값과 레이블을 분리하여 데이터셋 구성
input_dataset = combined_dataset.map(lambda x_4d, x_2d, x_3d: (x_4d, x_2d, x_3d))
# 출력값이 필요하지 않으므로 label_dataset은 사용하지 않음

# 모델 학습
model.fit(input_dataset, epochs=5)
