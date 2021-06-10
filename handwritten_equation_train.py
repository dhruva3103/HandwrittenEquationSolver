import pandas as pd
import numpy as np
import pickle
import keras
from keras.models import Model
from keras.layers import *
from keras import optimizers
from keras.layers import Dense, Input, Dropout, Flatten
from keras.models import Sequential
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.utils import np_utils
from keras.utils.np_utils import to_categorical
from tensorflow.keras import backend as k
from keras.models import model_from_json
k.set_image_data_format('channels_first')


df_train = pd.read_csv('/home/lp-n-12/Downloads/Handwritten-Equation-Solver-master/train_final.csv')
labels = df_train[['784']]

df_train.drop(df_train.columns[[784]], axis=1, inplace=True)
# print(df_train.head())

labels = np.array(labels)
cat = to_categorical(labels, num_classes=13)
# print(cat[0])

l = []
for i in range(47504):
    l.append(np.array(df_train[i:i+1]).reshape(28, 28, 1))

np.random.seed(7)

model = Sequential()
model.add(Conv2D(30, (5, 5), input_shape=(28, 28, 1), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(15, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(13, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(np.array(l), cat, epochs=10, batch_size=200, shuffle=True, verbose=1)

model_json = model.to_json()
with open("model_final_handwritten_equation.json", "w") as json_file:
    json_file.write(model_json)
model.save_weights("model_final_handwritten_equation.h5")