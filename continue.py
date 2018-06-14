import numpy as np
import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import glob
import random

from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from keras.utils import to_categorical


print ("Setting up data...")
# Set up the data
gestures = ['a', 'b', 'c', 'd', 'g', 'h', 'i', 'l', 'v', 'y']


class HandPics():
    """
    Used to store the entire dataset or subset of images, with methods to augment etc accordingly
    """
    def __init__(self):
        self.paths = []
        self.labels = []
        self.images = []
        self.final_labels = []
        
    def add_labels(self,new_labels):
        self.labels.extend(new_labels)
    
    def add_paths(self, new_paths):
        self.paths.extend(new_paths)
    
    def extract_and_augment(self):
        print ("Augmenting images...")
        rows = 128
        cols = 128

        for i in range(len(self.paths)):
            # Add the 14 labels
            self.final_labels.extend(([self.labels[i]] * 14))
            
            rotation = random.randint(10,20)
            shift = random.randint(10,20)

            R1 = cv2.getRotationMatrix2D((cols/2,rows/2),rotation,1)
            R2 = cv2.getRotationMatrix2D((cols/2,rows/2),rotation*-1,1)

            R_zoom = cv2.getRotationMatrix2D((cols/2,rows/2),rotation,1.5)
            R_out = cv2.getRotationMatrix2D((cols/2,rows/2),rotation,0.5)

            pts1 = np.float32([[30,30],[30,108],[108,30],[108,108]])
            pts2 = np.float32([[30-shift,30-shift],[30-shift,128],[98+shift,30+shift],[98+shift,98+shift]])
            pts3 = np.float32([[30+shift,30+shift],[30+shift,98+shift],[128,0],[128,128]])

            zoom_left = cv2.getPerspectiveTransform(pts1,pts2)
            zoom_right = cv2.getPerspectiveTransform(pts1,pts3)

            image = mpimg.imread(self.paths[i])

            # Flip
            self.images.append(image * 1./255 - 0.5)
            flip = (cv2.flip(image, 1))
            self.images.append(flip)

            # Rotate and zoom
            self.images.append(cv2.warpAffine(image,R1,(cols,rows)))
            self.images.append(cv2.warpAffine(image,R2,(cols,rows)))

            self.images.append(cv2.warpAffine(flip,R1,(cols,rows)))
            self.images.append(cv2.warpAffine(flip,R2,(cols,rows)))

            self.images.append(cv2.warpAffine(image,R_zoom,(cols,rows)))
            self.images.append(cv2.warpAffine(image,R_out,(cols,rows)))

            self.images.append(cv2.warpAffine(flip,R_zoom,(cols,rows)))
            self.images.append(cv2.warpAffine(flip,R_out,(cols,rows)))

            # Shear
            self.images.append(cv2.warpPerspective(image, zoom_left,(cols,rows)))
            self.images.append(cv2.warpPerspective(image, zoom_right,(cols,rows)))

            self.images.append(cv2.warpPerspective(flip, zoom_left,(cols,rows)))
            self.images.append(cv2.warpPerspective(flip, zoom_right,(cols,rows)))
        
        self.images = np.array(self.images)
        self.final_labels = np.array(self.final_labels)
    
    def encode_labels(self):
        print ("Encoding labels...")
        # Encode string labels into integer format
        enc = LabelEncoder()
        self.final_labels = enc.fit_transform(self.final_labels)

        # Convert to one-hot
        self.final_labels = to_categorical(self.final_labels, num_classes=None)
    
    def data(self):
        return self.images
    
    def y(self):
        return self.final_labels

        
plain_bg_data = HandPics()
real_bg_data = HandPics()

# Add filepaths to array
prefix = 'Triesch/*'
suffix = '.pgm'

# Plain background images
for char in gestures:
    label = None
    paths_char = []
    for i in range(1,3):
        if (char == 'h'):
            label = 'g'
        elif (char == 'l'):
            label = 'd'
        elif (char == 'i'):
            continue
        else:
            label = char
        paths_char.extend(glob.glob(prefix + char + str(i) + suffix))
    plain_bg_data.add_labels(([label] * len(paths_char)))
    plain_bg_data.add_paths(paths_char)


real_paths = []
real_labels = []
# Real background images
for char in gestures:
    label = None
    paths_char = []
    if (char == 'h'):
        label = 'g'
    elif (char == 'l'):
        label = 'd'
    elif (char == 'i'):
        continue
    else:
        label = char
    paths_char.extend(glob.glob(prefix + char + '3' + suffix))
    real_labels.extend(([label] * len(paths_char)))
    real_paths.extend(paths_char)

# Now split into train test sets
from sklearn.model_selection import train_test_split

train_real, val_real, train_real_y, val_real_y = train_test_split(real_paths, real_labels, test_size=0.4, random_state=42)


# Add the respective halves to real and plain, augment them and encode the labels
real_bg_data.add_paths(val_real)
real_bg_data.add_labels(val_real_y)

real_bg_data.extract_and_augment()
real_bg_data.encode_labels()

plain_bg_data.add_paths(train_real)
plain_bg_data.add_labels(train_real_y)

plain_bg_data.extract_and_augment()
plain_bg_data.encode_labels()

# Get the final training and validation datasets
X_train = plain_bg_data.data()
y_train = plain_bg_data.y()

X_val = real_bg_data.data()
y_val = real_bg_data.y()

# Create the model
from keras.models import Sequential
import keras.layers

model = Sequential()

model.add(keras.layers.Reshape(input_shape=(128, 128), target_shape=(128,128,1)))
model.add(keras.layers.Conv2D(input_shape=(128,128,1), filters=32, kernel_size=3, strides=1, activation='tanh', kernel_initializer='glorot_uniform'))
model.add(keras.layers.Conv2D(filters=32, kernel_size=3, strides=1, padding='same', activation="relu", kernel_initializer='he_normal'))
model.add(keras.layers.MaxPooling2D())

model.add(keras.layers.Conv2D(filters=64, kernel_size=3, strides=1, padding='same', activation="tanh", kernel_initializer='glorot_uniform'))
model.add(keras.layers.Conv2D(filters=64, kernel_size=3, strides=1, padding='same', activation="relu", kernel_initializer='he_normal'))
model.add(keras.layers.MaxPooling2D())

model.add(keras.layers.Conv2D(filters=128, kernel_size=3, strides=1, padding='same', activation="tanh", kernel_initializer='glorot_uniform'))
model.add(keras.layers.Conv2D(filters=128, kernel_size=3, strides=1, padding='same', activation="relu", kernel_initializer='he_normal'))
model.add(keras.layers.MaxPooling2D())

model.add(keras.layers.Conv2D(filters=256, kernel_size=3, strides=1, padding='same', activation="tanh", kernel_initializer='glorot_uniform'))
model.add(keras.layers.Conv2D(filters=256, kernel_size=3, strides=1, padding='same', activation="relu", kernel_initializer='he_normal'))
model.add(keras.layers.MaxPooling2D())

model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(32, activation='relu'))
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.Dense(7, activation='softmax'))

model.summary()

from keras.optimizers import Adam
adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])


# Create callback function for training, to save best models
from keras.callbacks import ModelCheckpoint, CSVLogger

save_model = ModelCheckpoint("shape-128-augmented-june14.h5", monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
csv_logger = CSVLogger('training.log')

# Train!
model.fit(X_train, y_train, epochs=1000, verbose=1, batch_size=64, validation_data=(X_val, y_val), shuffle=True, callbacks=[save_model, csv_logger], initial_epoch=1)