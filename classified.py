import cv2
import numpy as np
import copy
import math
import os
import requests
import operator

from keras.models import load_model
model = load_model('shape-128-grayscale-june11-withflip.h5')

gestures = {0: 'Closed fist', 
            1: 'Four fingers', 
            2: 'C shape', 
            3: 'Number One', 
            4: 'One finger pointing', 
            5: 'Pee', 
            6: 'Two beers', 
            7: 'Spidey'}


def classify(img):
    image = np.array([img * 1./255 - 0.5])
    out = model.predict(image)
    gesture = gestures[np.argmax(out)]
    if np.max(out) > 0.40:
        return gesture
    else:
        return "Not clear"

# parameters
threshold = 60  #  BINARY threshold
bgSubThreshold = 50

# variables
isBgCaptured = 0   # bool, whether the background captured
triggerSwitch = False  # if true, keyborad simulator works

def printThreshold(thr):
    print("! Changed threshold to "+str(thr))


def removeBG(frame):
    fgmask = bgModel.apply(frame, 0 ,0)
    # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    # res = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)

    kernel = np.ones((3, 3), np.uint8)
    fgmask = cv2.erode(fgmask, kernel, iterations=1)
    res = cv2.bitwise_and(frame, frame, mask=fgmask)
    return res


# Camera
camera = cv2.VideoCapture(0)
camera.set(10,200)

while camera.isOpened():
    ret, frame = camera.read()

    #frame = cv2.bilateralFilter(frame, 5, 50, 100)  # smoothing filter
    frame = cv2.flip(frame, 1)  # flip the frame horizontally

    # Draw the region of interest onto original frame for visual feedback
    cv2.rectangle(frame, ((frame.shape[1]-400), 0),
                 (frame.shape[1], 400), (255, 0, 0), 2)
    #  Main operation
    # this part will run when you press 'b' on a frame
    # Process Image
    #img = removeBG(frame)

    # extract the ROI
    roi = frame[0:400, (frame.shape[1]-400):frame.shape[1]]
    #roi = removeBG(roi)
    roi = cv2.resize(roi, (128,128), interpolation = cv2.INTER_AREA)
    
    # convert the image into binary image
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.flip(gray, 1)
    #gesture = classify(gray)

    thresh = np.dstack((gray, gray, gray))

    # Overlay
    frame[0:128, (frame.shape[1]-128):frame.shape[1]] = thresh
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, classify(gray), (10,500), font, 4,(0,0,255),4,cv2.LINE_AA)
    
    cv2.imshow('original', frame)


    k = cv2.waitKey(10)
    if k == 27:  # press ESC to exit
        camera.release()
        cv2.destroyAllWindows()
        break
