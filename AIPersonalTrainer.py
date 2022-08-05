import cv2
import numpy as np
import  time
import poseEstimationModule as pem

count = 0 
cap =  cv2.VideoCapture(0)
previous_time = 0

class directions():
    up = 1 
    down = 0
direction = directions.up

detector = pem.PoseEstimation()
def counter(img , left_right ):
    color = (255,0,0)
    global direction
    global count
    if left_right == "left":
        index = [11,13,15]
        min_range,max_range = 210 ,300
        min_bar ,max_bar = 450 ,200
        min_percentage ,max_percentage = 0 , 100
    else:
        index = [12,14,16]
        min_range,max_range = 50 ,150
        min_bar ,max_bar = 200 ,450
        min_percentage ,max_percentage = 100 , 0
    angle = detector.findAngle(img , index[0] , index[1] , index[2])
    percentge = np.interp(angle , (min_range ,max_range) , (min_percentage , max_percentage))
    bar = np.interp(angle , (min_range ,max_range) , (min_bar , max_bar))
    if percentge == 100 :
        color = (0 , 255 , 0)
        if direction == directions.down:
            count += 0.5
            direction = directions.up
    if percentge == 0 :
        color = (0 , 255 , 0)
        if direction == directions.up:
            count += 0.5
            direction = directions.down
    cv2.rectangle(img , (50 , 200) , (100 ,450) , color , 3)
    cv2.rectangle(img , (50 , int(bar)) , (100 ,450) , color , cv2.FILLED)
    cv2.putText(img , f'{int(percentge)}%' , (50 , 500) , cv2.FONT_HERSHEY_PLAIN , 3 , color , 3)
    cv2.putText(img , left_right , (30 , 190) , cv2.FONT_HERSHEY_PLAIN , 3 , color , 3)
    return img 

while True:
    success ,img = cap.read()
    img = cv2.resize(img, (500,500), interpolation = cv2.INTER_AREA)

    img = detector.findPoses(img)
    lmList = detector.findPosition(img ,False)
    if len(lmList) !=0:
        right_left_detector = detector.detect_right_left()
        color = (255 , 0 , 0)
        
        if right_left_detector == "left":
            img  = counter(img,right_left_detector)
        else:
            img  = counter(img,right_left_detector)
        cv2.rectangle(img , (0 , 0) , (150 ,150) , (0 , 255 , 0 ) , cv2.FILLED)
        if count <10:
            cv2.putText(img , str(int(count)) , (35 , 115) , cv2.FONT_HERSHEY_PLAIN , 8 , (255 , 0 , 0) , 8)
        else:
            cv2.putText(img , str(int(count)) , (5 , 115) , cv2.FONT_HERSHEY_PLAIN , 8 , (255 , 0 , 0) , 8)
    current_time = time.time() 
    fbs = 1 / (current_time - previous_time)
    previous_time = current_time
    cv2.putText(img, f'fbs: {int (fbs)}',(400,30), cv2.FONT_HERSHEY_PLAIN , 2,(255,0,0) , 2)
    cv2.imshow("Image" , img)
    cv2.waitKey(1)