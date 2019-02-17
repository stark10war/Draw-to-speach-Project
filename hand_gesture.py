# -*- coding: utf-8 -*-
"""
Created on Tue Feb 12 13:29:06 2019

@author: Shashank
"""
import numpy as np
import pandas as pd
import cv2 
import os
os.chdir('D:/Python practice/Hand Gesture')

url = 'http://192.168.0.106:8080/video'

cap = cv2.VideoCapture(0)
lower = np.array([36, 25, 25])
upper = np.array([70, 255,255])

paint = np.zeros((480,640,1), np.uint8)

old_pos = (0,0)

while True:s
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
    
    contrs,h =cv2.findContours(img_blur1.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)#Finding conturs
    
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
            if old_pos != (0,0) and diff<(50,50) :
                cv2.line(paint,old_pos,new_pos,(255,255,255),5)
                cv2.line(img,old_pos,new_pos,(255,255,255),5)
            old_pos = new_pos
            
    canvas = paint[rect_y:rect_y+rect_height,rect_x:rect_x+rect_len]
    #cv2.imshow('res', res)
    cv2.imshow('img', img)
    #cv2.imshow('img_blur1', img_blur1)
    cv2.imshow('canvas', canvas)
    cv2.imshow('paint',paint)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()





