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
    if (char == 'h'):
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

X_train, X_val, y_train, y_val = train_test_split(paths, y, test_size=0.2, random_state=42)

x_val = []
for path in X_val:
    x_val.append(mpimg.imread(path))
x_val = np.array(x_val)

# Initialize the train generator to generate augmented batches of data
from sklearn.utils import shuffle
rows = 128
cols = 128

def generator(paths, labels, zoom=True, warp=True, rotation=20):
    R1 = cv2.getRotationMatrix2D((cols/2,rows/2),rotation,1)
    R2 = cv2.getRotationMatrix2D((cols/2,rows/2),rotation*-1,1)
    
    R_zoom = cv2.getRotationMatrix2D((cols/2,rows/2),rotation,1.5)
    R_out = cv2.getRotationMatrix2D((cols/2,rows/2),rotation,0.5)
    
    pts1 = np.float32([[30,30],[30,108],[108,30],[108,108]])
    pts2 = np.float32([[0,0],[0,128],[100,50],[100,100]])
    pts3 = np.float32([[50,50],[50,100],[128,0],[128,128]])
                      
    zoom_left = cv2.getPerspectiveTransform(pts1,pts2)
    zoom_right = cv2.getPerspectiveTransform(pts1,pts3)
    
    # Batches of 14 images
    for i in range(len(paths)):
        # Initialize empty batch to be filled in
        batch_images = []
        batch_labels = np.array([labels[i]] * 14)

        image = mpimg.imread(paths[i])
    
        # Flip
        batch_images.append(image)
        flip = (cv2.flip(image, 1))
        batch_images.append(flip)
        
        # Rotate and zoom
        batch_images.append(cv2.warpAffine(image,R1,(cols,rows)))
        batch_images.append(cv2.warpAffine(image,R2,(cols,rows)))
        
        batch_images.append(cv2.warpAffine(flip,R1,(cols,rows)))
        batch_images.append(cv2.warpAffine(flip,R2,(cols,rows)))
        
        batch_images.append(cv2.warpAffine(image,R_zoom,(cols,rows)))
        batch_images.append(cv2.warpAffine(image,R_out,(cols,rows)))
        
        batch_images.append(cv2.warpAffine(flip,R_zoom,(cols,rows)))
        batch_images.append(cv2.warpAffine(flip,R_out,(cols,rows)))
        
        # Shear
        batch_images.append(cv2.warpPerspective(image, zoom_left,(cols,rows)))
        batch_images.append(cv2.warpPerspective(image, zoom_right,(cols,rows)))
        
        batch_images.append(cv2.warpPerspective(flip, zoom_left,(cols,rows)))
        batch_images.append(cv2.warpPerspective(flip, zoom_right,(cols,rows)))
        
        batch_images = np.array(batch_images)
        
        yield shuffle(batch_images, batch_labels)

train_generator = generator(X_train, y_train)


# Create the model
from keras.models import Sequential

from keras.models import load_model
model = load_model('shape-128-augmented-june13.h5')
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
model.fit_generator(generator=train_generator, steps_per_epoch=len(X_train), epochs=1, verbose=100, validation_data=(x_val, y_val), shuffle=True, callbacks=[save_model, save_session])
