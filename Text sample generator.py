# -*- coding: utf-8 -*-
"""
Created on Fri Feb 22 13:55:26 2019

@author: Shashank
"""

import numpy as np
import keyboard
import cv2 
import os
os.chdir('D:/Python practice/Hand Gesture')



def preproces_displayFrame(image):
    img  = image
    img =  cv2.flip(img,1)
#    cv2.rectangle(img, (0,0), (50, 50),(0,0,255),2)
    #cv2.putText(img,'Draw here: ', (rect_x,rect_y), cv2.FONT_HERSHEY_SIMPLEX,1, (255,0,0), 2)
    return img

def color_mask(image, lower_bound, upper_bound):
    img =  cv2.flip(image,1)
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(img_hsv,lower_bound,upper_bound)
    img_blur1 =  cv2.medianBlur(mask, 15)
    return img_blur1


def crop_maxContour(image):
    conts, h = cv2.findContours(image.copy(),cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    max_cont = max(conts, key = cv2.contourArea)
    x,y,w,h = cv2.boundingRect(max_cont)
    cropped  =  image[y:y+h,x:x+w]
    return cropped


## Setting the base directory
BASE_DIR = os.path.dirname(os.path.abspath('face-train.py'))
image_dir =  os.path.join(BASE_DIR, "Train_images")

character_name = input("Please Enter Character  name to generate its sample folder: ") ##Get input label from the user
os.mkdir(os.path.join(image_dir,character_name.lower()))

cap = cv2.VideoCapture(0)
lower = np.array([36, 25, 25])
upper = np.array([70, 255,255])
paint = np.zeros((480,640,1), np.uint8)
old_center =  (0,0)
text_count = 0
all_points = list()
while True:
    ret, img = cap.read()
    frame = preproces_displayFrame(img)
    green_mask = color_mask(img, lower, upper)
    contrs,h =cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if len(contrs) != 0:
        max_cont = max(contrs, key = cv2.contourArea)
        x,y,w,h = cv2.boundingRect(max_cont) #geting rectamgle cordinates
        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,255),2) # Draw rectangle on contures
        center = (int(x+w/2), int(y+h/2))
        cv2.circle(frame,center, 3, (0,0,255),-1)
             
        diff_x = abs(old_center[0] - center[0])
        diff_y = abs(old_center[1] - center[1])
        diff = (diff_x,diff_y)
        if old_center != (0,0) and diff<(60,60) and keyboard.is_pressed('shift'):
            cv2.line(paint,old_center,center,(255,255,255),3)
            all_points.append(center)
            for i in  range(len(all_points)-1):
                cv2.line(frame,all_points[i],all_points[i+1],(69,38,255),3)
            
        old_center = center    
    
    #Clear paint window
    if keyboard.is_pressed('c'):
        paint = np.zeros(paint.shape,np.uint8)
        
     # saving text images       
    if keyboard.is_pressed('shift') == False and paint.sum() > 255 and diff<(60,60):
        text_count = text_count+1
        text = crop_maxContour(paint)
        text = cv2.resize(text, (28,28))
        cv2.imwrite('Train_images/'+character_name+'/'+character_name+str(text_count)+ '.jpg', text)
        paint = np.zeros(paint.shape,np.uint8)
        all_points = list()
    

    
    cv2.putText(frame,'Total Samples Saved: '+str(text_count), (25,440), cv2.FONT_HERSHEY_SIMPLEX,1, (255,0,0), 2)       
    cv2.imshow('frame', frame)
    cv2.imshow('paint', paint)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
        

