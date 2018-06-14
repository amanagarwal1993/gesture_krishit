import numpy as np
import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import glob

from keras import backend as K
K.set_learning_phase(0)

import tensorflow as tf

# Set up the data
gestures = ['a', 'b', 'c', 'd', 'g', 'h', 'i', 'l', 'v', 'y']

y_all = []
paths = []

# Add filepaths to array
prefix = 'Triesch/*'
suffix = '.pgm'

for char in gestures:
    paths_char = []
    for i in range(1,4):
        paths_char.extend(glob.glob(prefix + char + str(i) + suffix))
    if (char == 'h' or char == 'i'):
        char = 'g'
    elif (char == 'l'):
        char = 'd'
    y_all.extend(([char] * len(paths_char)))
    paths.extend(paths_char)

paths = np.array(paths)

from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from keras.utils import to_categorical

# Encode string labels into integer format
enc = LabelEncoder()
y_all = enc.fit_transform(y_all)

# Convert to one-hot
y = to_categorical(y_all, num_classes=None)

from sklearn.model_selection import train_test_split

X_all = []
for path in X_all:
    X_all.append(mpimg.imread(path))
X_aal = np.array(X_all)

X_train, X_val, y_train, y_val = train_test_split(X_all, y, test_size=0.2, random_state=42)

# Create the model
from keras.models import Sequential
import keras.layers

model = Sequential()

model.add(keras.layers.Reshape(input_shape=(128, 128), target_shape=(128,128,1)))
model.add(keras.layers.Conv2D(input_shape=(128,128,1), filters=32, kernel_size=3, strides=1, activation='tanh', kernel_initializer='glorot_uniform'))
model.add(keras.layers.Conv2D(filters=32, kernel_size=3, strides=1, padding='same', activation="tanh", kernel_initializer='glorot_uniform'))
model.add(keras.layers.MaxPooling2D())

model.add(keras.layers.Conv2D(filters=64, kernel_size=3, strides=1, padding='same', activation="tanh", kernel_initializer='glorot_uniform'))
model.add(keras.layers.Conv2D(filters=64, kernel_size=3, strides=1, padding='same', activation="tanh", kernel_initializer='glorot_uniform'))
model.add(keras.layers.MaxPooling2D())

model.add(keras.layers.Conv2D(filters=128, kernel_size=3, strides=1, padding='same', activation="tanh", kernel_initializer='glorot_uniform'))
model.add(keras.layers.Conv2D(filters=128, kernel_size=3, strides=1, padding='same', activation="tanh", kernel_initializer='glorot_uniform'))
model.add(keras.layers.MaxPooling2D())

model.add(keras.layers.Conv2D(filters=256, kernel_size=3, strides=1, padding='same', activation="tanh", kernel_initializer='glorot_uniform'))
model.add(keras.layers.Conv2D(filters=256, kernel_size=3, strides=1, padding='same', activation="tanh", kernel_initializer='glorot_uniform'))
model.add(keras.layers.MaxPooling2D())

model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(32, activation='relu'))
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.Dense(7, activation='softmax'))

model.summary()

model.compile(loss='categorical_crossentropy', optimizer="sgd", metrics=['accuracy'])


# Create callback functions for training, to save best models
from keras.callbacks import ModelCheckpoint, Callback

save_model = keras.callbacks.ModelCheckpoint("shape-128-augmented-june13.h5", monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)

class SaveSession(Callback):
    def on_train_begin(self, logs={}):
        self.best = 0
        self.saver = tf.train.Saver()
    
    def on_epoch_end(self, logs={}):
        acc = logs.get('val_acc')
        if (acc >= self.best):
            print(model.output.op.name)
            self.saver.save(K.get_session(), '/keras_model.ckpt')

save_session = SaveSession()


# Train!
model.fit(X_train, y_train epochs=100, verbose=1, validation_data=(X_val, y_val), shuffle=True, callbacks=[save_model, save_session])