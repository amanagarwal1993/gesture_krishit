from __future__ import print_function
#from azure.cognitiveservices.vision.customvision.training import training_api
#from azure.cognitiveservices.vision.customvision.training.models import ImageUrlCreateEntry


import cv2
import numpy as np
import copy
import math
import os
import requests
import operator


# Replace with a valid key
training_key = "[]"
prediction_key = "[]"


# Variables
_url = '[]'

_maxNumRetries = 10



def classify():
        data= None
        
        # Change path
        pathToFileInDisk = r'C:\Users\Krishit Arora\Desktop\1.jpg'
        imgt=cv2.imread(pathToFileInDisk,0)
        
        #cv2.imshow('image', imgt)
        
        # Computer Vision parameters
        params = { 'visualFeatures' : 'Color,Categories'} 
        
        headers = dict()
        headers['Prediction-Key'] = prediction_key
        headers['Content-Type'] = 'application/octet-stream'
        
        response   = requests.post(_url, 
                                   headers=headers, 
                                   params=params, 
                                   data=data)
        
        response.raise_for_status()
        
        analysis      = response.json()
        print(str(analysis["Predictions"][0]["Tag"])+'\t'+str(analysis["Predictions"][0]["Probability"]))
        cv2.imshow(str(analysis["Predictions"][0]["Tag"]),imgt)
        print (analysis)
        

# parameters
cap_region_x_begin=0.5  # start point/total width
cap_region_y_end=0.8  # start point/total width
threshold = 60  #  BINARY threshold
blurValue = 41  # GaussianBlur parameter
bgSubThreshold = 50

# variables
isBgCaptured = 0   # bool, whether the background captured
triggerSwitch = False  # if true, keyborad simulator works
i=0

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


def calculateFingers(res,drawing):  # -> finished bool, cnt: finger count
    #  convexity defect
    hull = cv2.convexHull(res, returnPoints=False)
    if len(hull) > 3:
        defects = cv2.convexityDefects(res, hull)
        if type(defects) != type(None):  # avoid crashing.   (BUG not found)

            cnt = 0
            for i in range(defects.shape[0]):  # calculate the angle
                s, e, f, d = defects[i][0]
                start = tuple(res[s][0])
                end = tuple(res[e][0])
                far = tuple(res[f][0])
                a = math.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
                b = math.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
                c = math.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)
                angle = math.acos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c))  # cosine theorem
                if angle <= math.pi / 2:  # angle less than 90 degree, treat as fingers
                    cnt += 1
                    cv2.circle(drawing, far, 8, [211, 84, 0], -1)
            return True, cnt
    return False, 0


# Camera
camera = cv2.VideoCapture(0)
camera.set(10,200)
cv2.namedWindow('trackbar')
cv2.createTrackbar('trh1', 'trackbar', threshold, 100, printThreshold)


while i<1:
    while camera.isOpened():
        ret, frame = camera.read()
        
        threshold = cv2.getTrackbarPos('trh1', 'trackbar')
        
        frame = cv2.bilateralFilter(frame, 5, 50, 100)  # smoothing filter
        frame = cv2.flip(frame, 1)  # flip the frame horizontally
        
        # Draw the region of interest onto original frame for visual feedback
        cv2.rectangle(frame, (int(cap_region_x_begin * frame.shape[1]), 0),
                     (frame.shape[1], int(cap_region_y_end * frame.shape[0])), (255, 0, 0), 2)
        if isBgCaptured == 0:
            cv2.imshow('original', frame)

        #  Main operation
        # this part will run when you press 'b' on a frame
        if isBgCaptured == 1:
            # Process Image
            #img = removeBG(frame)
            
            # extract the ROI
            roi = frame[0:int(cap_region_y_end * frame.shape[0]),
                        int(cap_region_x_begin * frame.shape[1]):frame.shape[1]]
            
            roi = removeBG(roi)
            #cv2.imshow('mask', img)
            # convert the image into binary image
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            #cv2.imshow('gray', gray)
            #blur = cv2.GaussianBlur(gray, (blurValue, blurValue), 0)
            #cv2.imshow('blur', blur)
            #ret, thresh = cv2.threshold(blur, threshold, 255, cv2.THRESH_BINARY)
            
            thresh = np.dstack((gray, gray, gray))
            
            # Overlay
            frame[0:int(cap_region_y_end * frame.shape[0]),int(cap_region_x_begin * frame.shape[1]):frame.shape[1]] = thresh
            
            cv2.imshow('original', frame)
            
            
        # Keyboard OP
        k = cv2.waitKey(10)
        if k == 27:  # press ESC to exit
            cv2.destroyAllWindows()
            break
        
        if k == ord('b'):  # press 'b' to capture the background
            bgModel = cv2.createBackgroundSubtractorMOG2(0, bgSubThreshold)
            isBgCaptured = 1
            print ('!!!Background Captured!!!')
            i=i+1

        #elif k == ord('c'):
            #img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            #cv2.imwrite('1.jpg', img_gray)
            #classify()        

        elif k == ord('r'):  # press 'r' to reset the background
            bgModel = None
            triggerSwitch = False
            isBgCaptured = 0
            i=0
            print ('!!!Reset BackGround!!!')
            
        elif k == ord('n'):
            triggerSwitch = True
            print ('!!!Trigger On!!!')

