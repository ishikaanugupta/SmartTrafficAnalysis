import cv2
import numpy as np
import matplotlib.pyplot as plt 
from tracker import *

tracker = EuclideanDistanceTracker() #Create tracker object

object_detector = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=40)

val_dict={}

def eqn_of_line(x1, y1, x2, y2):   # line of the form ax + by + c = 0
    a = y2 - y1
    b = x1 - x2
    c = y1*x2 - x1*y2
    return a,b,c

def add_element(dict,key,value):
    if key not in dict:
        dict[key]=[]
    dict[key].append(value)

counter = 0
frame_counter = 0

video = cv2.VideoCapture("3.mp4")

while True:
    ret, orig_frame = video.read()
    #counter=0
    frame_counter += 1
    if not ret:
        video = cv2.VideoCapture("3.mp4")
        continue
    
    frame = orig_frame[310:700, 650: 1100]
    frame = cv2.GaussianBlur(frame, (5,5), 0)
    senstivity = 33
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_limit = np.array([0,0,102])
    upper_limit = np.array([179,255-senstivity,255])
    mask = cv2.inRange(hsv, lower_limit, upper_limit)
    edges = cv2.Canny(mask, 75, 150)
    
    lines = cv2.HoughLinesP(edges, 0.02, np.pi/180, threshold = 1, minLineLength = 10, maxLineGap = 1)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            cv2.line(frame, (x1,y1), (x2,y2), (0,255,0), 5)
            
    mask = object_detector.apply(frame)
    _,mask = cv2.threshold(mask, 254, 255, cv2.THRESH_BINARY)
    contours,_ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    detections = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 1000:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 3)
            detections.append([x,y,w,h])
            
            
    #Object Tracking
    boxes_ids = tracker.update(detections)
    for box_id in boxes_ids:
        x,y,w,h,id = box_id     #x,y coordinates of top-left corner; w is width; h is height
        cv2.putText(frame, str(id), (x,y-15), cv2.FONT_HERSHEY_PLAIN, 2, (255,0,0), 2)
        #cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 3)
        x_c = w//2
        y_c = h//2
        cx = x + x_c
        cy = y + y_c
        #put the 5th frame condition
        #if(frame_counter%5 == 0):
            if lines is not None:
                for line in lines:
                    x1, y1, x2, y2 = line.reshape(4)
                    a,b,c = eqn_of_line(x1, y1, x2, y2)
        if a != 9999999:
            if cx*a + cy*b +c == 0:
                counter += 1
            add_element(val_dict,frame,counter)
        if max(val_dict[vid]) > lane_threshold:
            parameter = "FreqLaneChange"
            img = cv2.rectangle(frame, (x,y), (x+w,y+h), (0, 0, 255),2)
            cv2.putText(frame, 'RD', (cx, cy), font, 0.3, (0, 0, 255), 1, cv2.LINE_AA)
                                        
    
    cv2.imshow("Frame",frame)
    print("Object ID:",id, "Counter value of object ID:",counter)
    print("Frame Count",frame_counter)
     
        
    key = cv2.waitKey(30)
    if key == 27:
        break
            
video.release()
cv2.destroyAllWindows()

