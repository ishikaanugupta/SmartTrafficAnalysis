import cv2
import numpy as np
import matplotlib.pyplot as plt 
import trackerEuc
import vehicles
import time, datetime
import os

global w, h, frameArea, areaTH, alert_records_path
global font, cars, max_p_age, pid, cnt_up, cnt_down, cnt_wrong_up, cnt_wrong_down
global a, b, c, line_up, line_down, up_limit, down_limit, line_down_color, line_up_color
global pt1, pt2, pt3, pt4, pt5, pt6, pt7, pt8, pts_L1, pts_L2, pts_L3, pts_L4
global car_id, start_time, ax1, ax2, bx1, bx2, ay, by, RD, w, h
global tracker, object_detector, fgbg, kernalOp, kernalOp, kernalOp2


video = cv2.VideoCapture("Videos\\3.mp4")
# video = cv2.VideoCapture('Test model\\v7.mp4')
alert_records_path = 'Alerts'
w = video.get(3)
h = video.get(4)
frameArea = h*w
areaTH = frameArea/400

RD = []

a, b, c = 9999999, 9999999, 9999999

font = cv2.FONT_HERSHEY_SIMPLEX
cars = []
max_p_age = 5
pid = 1
cnt_up=0
cnt_down=0
cnt_wrong_up = 0
cnt_wrong_down = 0
line_up=int(3.5*(h/5))
line_down=int(4*(h/5))
up_limit=int(3*(h/5))
down_limit=int(4.5*(h/5))
line_down_color=(255,0,0)
line_up_color=(255,0,255)
pt1 =  [0, line_down]
pt2 =  [w, line_down]
pts_L1 = np.array([pt1,pt2], np.int32)
pts_L1 = pts_L1.reshape((-1,1,2))
pt3 =  [0, line_up]
pt4 =  [w, line_up]
pts_L2 = np.array([pt3,pt4], np.int32)
pts_L2 = pts_L2.reshape((-1,1,2))
pt5 =  [0, up_limit]
pt6 =  [w, up_limit]
pts_L3 = np.array([pt5,pt6], np.int32)
pts_L3 = pts_L3.reshape((-1,1,2))
pt7 =  [0, down_limit]
pt8 =  [w, down_limit]
pts_L4 = np.array([pt7,pt8], np.int32)
pts_L4 = pts_L4.reshape((-1,1,2))

car_id = 1
start_time = time.time()
#line a
ax1=0
ay=int(3*h/5)
ax2=w
#line b
bx1=0
by=ay+100
bx2=w

tracker = trackerEuc.EuclideanDistanceTracker()
object_detector = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=40)

fgbg = cv2.createBackgroundSubtractorMOG2(detectShadows=True)
kernalOp = np.ones((3,3),np.uint8)
kernalOp2 = np.ones((5,5),np.uint8)
kernalCl = np.ones((11,11),np.uint8)

def eqn_of_line(x1, y1, x2, y2):       
    a = y2 - y1
    b = x1 - x2
    c = y1*x2 - x1*y2
    return a, b, c

def Speed_Cal(time):
    global fixed_dist
    try:
        Speed = (fixed_dist*3600)/(time*1000)
        return Speed
    except ZeroDivisionError:
        print (5)


def lane_change(frame, frame_counter, lane_counter):

    global object_detector, tracker, a, b, c, lane_threshold
    
    frame = frame[310:700, 650: 1100]
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
        x, y, w, h, vid = box_id     #x, y coordinates of top-left corner; w is width; h is height
        cv2.putText(frame, str(vid), (x,y-15), cv2.FONT_HERSHEY_PLAIN, 2, (255,0,0), 2)
        # cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 3)
        x_c = w//2
        y_c = h//2
        cx = x + x_c
        cy = y + y_c
        #put the 5th frame condition
        if(frame_counter%5 == 0) or (frame_counter == 1):
            if lines is not None:
                for line in lines:
                    x1, y1, x2, y2 = line.reshape(4)
                    a, b, c = eqn_of_line(x1, y1, x2, y2)
        if a != 9999999:
            if cx*a + cy*b +c == 0:
                lane_counter += 1
        if lane_counter > lane_threshold:
            parameter = "FreqLaneChange"
            value = str(lane_counter)
            record_RD(frame, vid, time.time(), parameter, value)
            img = cv2.rectangle(frame, (x,y), (x+w,y+h), (0, 0, 255),2)
            cv2.putText(frame, 'RD', (cx, cy), font, 0.3, (0, 0, 255), 1, cv2.LINE_AA)
                                        
            print("Object ID:", vid, "Counter value of object ID:",lane_counter)

    return frame, frame_counter, lane_counter

def wrong_direction_speed(frame):
    
    global fgbg, kernalOp, kernalCl, areaTH, up_limit, down_limit, line_up, line_down
    global cnt_up, cnt_down, cnt_wrong_up, cnt_wrong_down, line_up_color, line_down_color
    global font, cars, max_p_age, pid, speed_threshold
    global car_id, start_time, ax1, ax2, ay, bx1, bx2, by
    global a, b, c, frame_counter

    lane_frame = frame.copy()

    for i in cars:
        i.age_one()
    fgmask = fgbg.apply(frame)
    fgmask2 = fgbg.apply(frame)

    #Binarization
    ret, imBin = cv2.threshold(fgmask,200,255,cv2.THRESH_BINARY)
    ret, imBin2 = cv2.threshold(fgmask2,200,255,cv2.THRESH_BINARY)
    #OPening i.e First Erode the dilate
    mask = cv2.morphologyEx(imBin,cv2.MORPH_OPEN,kernalOp)
    mask2 = cv2.morphologyEx(imBin2,cv2.MORPH_CLOSE,kernalOp)
    #Closing i.e First Dilate then Erode
    mask = cv2.morphologyEx(mask,cv2.MORPH_CLOSE,kernalCl)
    mask2 = cv2.morphologyEx(mask2,cv2.MORPH_CLOSE,kernalCl)
    # dimesnions of entire frame
    height, width, channels = frame.shape

    #Find Contours
    countours0, hierarchy = cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    for cnt in countours0:
        area = cv2.contourArea(cnt)
        # print(area)
        if area > areaTH:
            ####Tracking######
            m = cv2.moments(cnt)
            cx = int(m['m10']/m['m00'])
            cy = int(m['m01']/m['m00'])
            x, y, w, h = cv2.boundingRect(cnt)

            new = True

            if cy in range(up_limit,down_limit):
                for i in cars:                    
                    if abs(x - i.getX()) <= w and abs(y - i.getY()) <= h:
                        new = False
                        i.updateCoords(cx, cy)

                        # SPEED CALCULATIONS PART
                        # speed part for right side
                        if cx > (width/2 + 50):
                            while int(line_up) == int((y+y+h)/2):
                                start_time = time.time()
                                break
                                
                            while int(line_up) <= int((y+y+h)/2):
                                if int(line_down) <= int((y+y+h)/2) & int(line_down+10) >= int((y+y+h)/2):
                                    Speed = Speed_Cal(time.time() - start_time)
                                    if Speed != None and Speed > speed_threshold:
                                        RD.append(i.getId())
                                        parameter = "Overspeeding"
                                        value = str(Speed)
                                        img = cv2.rectangle(frame, (x,y), (x+w,y+h), (0, 0, 255),2)
                                        cv2.putText(frame, 'RD', (i.getX(), i.getY()), font, 0.3, (0, 0, 255), 1, cv2.LINE_AA)
                                        record_RD(frame, i.getId(), time.time(), parameter, value)
                                    print("Car ID: "+str(i.getId())+" Speed: "+str(Speed)+" KM/H")
                                    cv2.putText(frame, "Speed: "+str(Speed)+"KM/H", (x,y-15), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0),3)
                                    break
                                else:
                                    cv2.putText(frame, "Calculating", (int(width),int(height/10)), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,255,0),3)
                                    break
                        # speed part for left side
                        elif cx < (width/2 - 100):
                            while int(line_down) == int((y+y+h)/2):
                                start_time = time.time()
                                break
                                
                            while int(line_down) >= int((y+y+h)/2):
                                if int(line_up) >= int((y+y+h)/2) & int(line_up-10) >= int((y+y+h)/2):
                                    Speed = Speed_Cal(time.time() - start_time)
                                    if Speed != None and Speed > speed_threshold:
                                        parameter = "Overspeeding"
                                        value = str(Speed)
                                        img = cv2.rectangle(frame, (x,y), (x+w,y+h), (0, 0, 255),2)
                                        cv2.putText(frame, 'RD', (i.getX(), i.getY()), font, 0.3, (0, 0, 255), 1, cv2.LINE_AA)
                                        record_RD(frame, i.getId(), time.time(), parameter, value)
                                    print("Car ID: "+str(i.getId())+" Speed: "+str(Speed)+" KM/H")
                                    cv2.putText(frame, "Speed: "+str(Speed)+"KM/H", (x,y-15), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,255), 3)
                                    break
                                else:
                                    cv2.putText(frame, "Calculating", (int(width),int(height/10)), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,255,0),3)
                                    break

                        # WRONG DIRECTION PART
                        # vehicles going UP
                        if i.going_UP(line_down,line_up)==True:
                            # right side
                            if cx > (width/2 + 50):
                                cnt_wrong_up += 1
                                parameter = "WrongDirection"
                                value = "UP"
                                record_RD(frame, i.getId(), time.time(), parameter, value)
                                img = cv2.rectangle(frame, (x,y), (x+w,y+h), (0, 0, 255),2)
                                cv2.putText(frame, 'RD', (i.getX(), i.getY()), font, 0.3, (0, 0, 255), 1, cv2.LINE_AA)        
                                print("ID:",i.getId(),'wrongly going up at', time.strftime("%c"), 'location', cx, cy)
                            # left side
                            else:
                                cnt_up+=1
                                print("ID:",i.getId(),'crossed going up at', time.strftime("%c"))
                            
                            
                        # vehicles going DOWN
                        elif i.going_DOWN(line_down,line_up)==True:
                            # left side
                            if cx < (width/2 - 50):
                                cnt_wrong_down += 1
                                parameter = "WrongDirection"
                                value = "DOWN"
                                record_RD(frame, i.getId(), time.time(), parameter, value)
                                img = cv2.rectangle(frame, (x,y), (x+w,y+h), (0, 0, 255),2)
                                cv2.putText(frame, 'RD', (i.getX(), i.getY()), font, 0.3, (0, 0, 255), 1, cv2.LINE_AA)
                                print("ID:",i.getId(),'wrongly going down at', time.strftime("%c"), 'location', cx, cy)
                            # right side
                            else:
                                cnt_down+=1
                                print("ID:", i.getId(), 'crossed going down at', time.strftime("%c"))
                        break

                    if i.getState()=='1':
                        if i.getDir()=='down'and i.getY() > down_limit:
                            i.setDone()
                        elif i.getDir()=='up'and i.getY() < up_limit:
                            i.setDone()
                            
                    if i.timedOut():
                        index=cars.index(i)
                        cars.pop(index)
                        del i

                if new==True: #If nothing is detected,create new
                    p = vehicles.Car(pid, cx, cy, max_p_age)
                    cars.append(p)
                    pid += 1

            cv2.circle(frame,(cx,cy),5,(0,0,255),-1)
            # img = cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)

    for i in cars:
        cv2.putText(frame, str(i.getId()), (i.getX(), i.getY()), font, 0.3, (255, 255, 255), 1, cv2.LINE_AA)

    str_up = 'CORRECT: '+str(cnt_up+cnt_down)
    str_down = 'WRONG DIRECTION: '+str(cnt_wrong_up+cnt_wrong_down)
    frame = cv2.polylines(frame,[pts_L1],False,line_down_color,thickness=2)
    frame = cv2.polylines(frame,[pts_L2],False,line_up_color,thickness=2)
    frame = cv2.polylines(frame,[pts_L3],False,(255,255,255),thickness=1)
    frame = cv2.polylines(frame,[pts_L4],False,(255,255,255),thickness=1)
    cv2.putText(frame, str_up, (10, 40), font, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, str_up, (10, 40), font, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, str_down, (10, 90), font, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, str_down, (10, 90), font, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

    return frame

def record_RD(frame, car_id, time, parameter, value):
    global alert_records_path

    timestamp = datetime.datetime.now()
    img_name = str(timestamp.strftime("%d_%m_%Y_%H_%M_%S")) + '__ID' + str(car_id) + '__' + str(parameter) + '.jpg'
    cv2.imwrite(os.path.join(alert_records_path, img_name), frame)

    ref = db.reference("/Alerts/Highway")

    ref.child(img_name).set(
        {
            "VehicleID": car_id,
            "Parameter": parameter,
            "Value": value,
            "Date": datetime.datetime.now(),
            "Time": datetime.datetime.now()
        }
    )

    print("\nRash driving detected! Car ID:", str(car_id), "Parameter:", parameter, "Vehicle value:", value, '\n')


# main video processing loop
frame_counter = 0
lane_counter = 0
lane_threshold = 3
speed_threshold = 20        # km/h
fixed_dist = 9.144          # used for calculating speed

while (video.isOpened()):
    ret, vid_frame = video.read()
    if ret:
        frame_counter += 1
        # processing w.r.t lane line detection
        lane_frame, frame_counter, lane_counter = lane_change(vid_frame, frame_counter, lane_counter)
        cv2.imshow('Lane change detection', lane_frame)
        
        # processing w.r.t speed & wrong direction detection
        dir_speed_frame = wrong_direction_speed(vid_frame)
        cv2.imshow('Speed & Wrong direction detection', dir_speed_frame)
        
        if cv2.waitKey(1) & 0xff==ord('q'):
            break
    else:
        break

video.release()
cv2.destroyAllWindows()


