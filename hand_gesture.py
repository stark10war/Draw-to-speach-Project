# -*- coding: utf-8 -*-
"""
Created on Tue Feb 12 13:29:06 2019

@author: Shashank
"""
import numpy as np
import keras
import keyboard
import cv2 
import os
os.chdir('D:/Python practice/Hand Gesture')

url = 'http://192.168.0.106:8080/video'

cap = cv2.VideoCapture(0)
lower = np.array([36, 25, 25])
upper = np.array([70, 255,255])

paint = np.zeros((480,640,1), np.uint8)

old_pos = (0,0)
img_count = 1 
while True:
    ret, img = cap.read()
    img =  cv2.flip(img,1)
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(img_hsv,lower,upper)
    #res = cv2.bitwise_and(img, img, mask = mask)
    img_blur1 =  cv2.medianBlur(mask, 15)
    rect_x = int(img.shape[1]/4)
    rect_y = int(img.shape[0]/4)*2
    rect_len =int(img.shape[1]/4)*2
    rect_height =int(img.shape[0]/4)*2
    cv2.rectangle(img, (rect_x,rect_y), (rect_x+rect_len, rect_y+rect_height),(0,0,255),2)
    cv2.putText(img,'Draw here: ', (rect_x,rect_y), cv2.FONT_HERSHEY_SIMPLEX,1, (255,0,0), 2)
    contrs,h =cv2.findContours(img_blur1.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)#Finding conturs
    canvas = paint[rect_y:rect_y+rect_height,rect_x:rect_x+rect_len]
    if len(contrs)==1:
        for i in range(len(contrs)):
            x,y,w,h = cv2.boundingRect(contrs[i]) #geting rectamgle cordinates
            cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0),2) # Draw rectangle on contures
            new_pos = (int(x+w/2), int(y+h/2))
            diff_x = abs(old_pos[0] - new_pos[0])
            diff_y = abs(old_pos[1] - new_pos[1])
            diff = (diff_x,diff_y)
            cv2.circle(img,new_pos, 5, (0,0,255),-1)
            #Drawing line between old frame center and current frame center
            if old_pos != (0,0) and diff<(50,50) and keyboard.is_pressed('shift') :
                cv2.line(paint,old_pos,new_pos,(255,255,255),5)   
            old_pos = new_pos
            
    model = keras.models.load_model("keras_alphanumeric.mod")
    canvas1 = cv2.resize(canvas, (32,32))
    canvas1 = canvas1.reshape((1,32,32,1))
    
    prediction =model.predict(canvas1)
        
 # saving text images       
#    if keyboard.is_pressed('shift') == False and canvas.sum() > 255:
 #       cv2.imwrite('canvas'+str(img_count)+ '.jpg', canvas)
  #      img_count = img_count+1
  

 #clearing the canvas
    if keyboard.is_pressed('c'):
        paint = np.zeros(paint.shape,np.uint8)
        
    #img[rect_y:rect_y+rect_height,rect_x:rect_x+rect_len] =  canvas
    #cv2.imshow('res', res)
    cv2.imshow('img', img)
    #cv2.imshow('img_blur1', img_blur1)
    cv2.imshow('canvas', canvas)
    #cv2.imshow('paint',paint)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()





