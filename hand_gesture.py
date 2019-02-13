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

url = 'http:192.168.10.42:8080/video'

cap = cv2.VideoCapture(url)
lower = np.array([36, 25, 25])
upper = np.array([70, 255,255])
while True:
    ret, img = cap.read()
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(img_hsv,lower,upper)
    #res = cv2.bitwise_and(img, img, mask = mask)
    img_blur1 =  cv2.medianBlur(mask, 15)
    #cv2.imshow('mask', mask)
    #cv2.imshow('img_blur', img_blur)
    
    contrs,h =cv2.findContours(img_blur1.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for i in range(len(contrs)):
        x,y,w,h = cv2.boundingRect(contrs[i])
        cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0),2)
    #cv2.imshow('res', res)
    cv2.imshow('img', img)
    cv2.imshow('img_blur1', img_blur1)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()





