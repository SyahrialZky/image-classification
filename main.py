import os
from PIL import Image, ImageOps
import numpy as np
import tensorflow as tf

dataset_dir = '../dataset'

rock_dir = os.path.join(dataset_dir, 'rock')
paper_dir = os.path.join(dataset_dir, 'paper')
scissors_dir = os.path.join(dataset_dir, 'scissors')

# pixeling & convert to grayscale


def imageToArray(filename):
    image = Image.open(filename)
    image_grayscale = ImageOps.grayscale(image)
    image_grayscale = image_grayscale.resize(size=(150, 150))
    image_array = np.array(image_grayscale)
    # normalization
    image_array = image_array / 255.0

    list_full = []
    for single_list in image_array:
        list_full.extend(single_list)
    return np.array(list_full)


data = []
label = []
# Labeling Model untuk rock [1, 0, 0]
for filename in os.listdir(rock_dir):
    image_path = os.path.join(rock_dir, filename)
    image_array = imageToArray(image_path)
    data.append(image_array)
    label.append([1, 0, 0])

# Labeling Model untuk paper [0, 0, 1]
for filename in os.listdir(paper_dir):
    image_path = os.path.join(paper_dir, filename)
    image_array = imageToArray(image_path)
    data.append(image_array)
    label.append([0, 0, 1])

# Labeling Model untuk scissors [0, 1, 0]
for filename in os.listdir(scissors_dir):
    image_path = os.path.join(scissors_dir, filename)
    image_array = imageToArray(image_path)
    data.append(image_array)
    label.append([0, 1, 0])

data = np.array(data)
label = np.array(label)


model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(64, input_shape=[22500], activation='relu'),
    tf.keras.layers.Dense(3, activation='softmax')
])

model.compile(
    metrics=['acc'],
    loss='categorical_crossentropy',
    optimizer='adam'
)

model.fit(
    data, label, epochs=10
)


def predict_image(image_path):
    test_array = imageToArray(image_path)
    prediction = model.predict(np.array([test_array]))
    class_names = ['Batu', 'Gunting', 'Kertas']
    predicted_class = class_names[np.argmax(prediction)]
    return predicted_class


test_image = 'test.png'
print(
    f'Prediksi Gambar : {predict_image(test_image)}')
