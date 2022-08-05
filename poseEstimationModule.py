from turtle import position
import cv2
import mediapipe as mp
import time
import math 

class PoseEstimation():
    def __init__(self, mode=False, complexity=1, smooth=True, segmentation = False, smooth_segmentation = True ,detectionCon=0.5, trackCon=0.5):

        self.mode = mode
        self.complexity = complexity
        self.segmentation = segmentation
        self.smooth_segmentation = smooth_segmentation
        self.smooth = smooth
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpDraw = mp.solutions.drawing_utils
        self.pose_module = mp.solutions.pose
        self.pose_object = self.pose_module.Pose(self.mode, self.complexity, self.smooth, self.segmentation, self.smooth_segmentation, self.detectionCon, self.trackCon)

    def findPoses(self,img,draw =True):
        imgRGB = cv2.cvtColor(img , cv2.COLOR_BGR2RGB)
        self.results =    self.pose_object.process(imgRGB)
        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(img,self.results.pose_landmarks, self.pose_module.POSE_CONNECTIONS)
        return img
    def findPosition(self,img,draw= True):
        self.landmarks_list = []
        if self.results.pose_landmarks:
            for id,lm in enumerate(self.results.pose_landmarks.landmark):
                height,width,channel = img.shape
                cx , cy = int (lm.x*width) ,int (lm.y*height)
                self.landmarks_list.append([id ,cx,cy ,lm.z])
                if draw:
                    cv2.circle(img,(cx,cy),5 , (255,0,0) ,cv2.FILLED)
        return self.landmarks_list

    def findAngle(self,img, p1,p2 ,p3 , draw =True):
        x1 , y1 = self.landmarks_list[p1][1:3]
        x2 , y2 = self.landmarks_list[p2][1:3]
        x3 , y3 = self.landmarks_list[p3][1:3]

        angle = math.degrees(math.atan2(y3-y2 , x3 -x2) - math.atan2(y1 -y2 ,x1-x2))
        
        if angle < 0 :
            angle += 360
        if draw:
            cv2.line(img, (x1,y1) ,(x2,y2) , (0 , 255 ,0) , 3)
            cv2.line(img, (x3,y3) ,(x2,y2) , (0 , 255 ,0) , 3)
            cv2.circle(img,(x1,y1),15 , (0,0,255) ,2)
            cv2.circle(img,(x2,y2),15 , (0,0,255) ,2)
            cv2.circle(img,(x3,y3),15 , (0,0,255) ,2)
            cv2.circle(img,(x1,y1),10 , (0,0,255) ,cv2.FILLED)
            cv2.circle(img,(x2,y2),10 , (0,0,255) ,cv2.FILLED)
            cv2.circle(img,(x3,y3),10 , (0,0,255) ,cv2.FILLED)
            cv2.putText(img,str(int(angle)),(x2 - 50 ,y2 +50) ,cv2.FONT_HERSHEY_PLAIN , 3 , (255 ,0,255) , 3)
        return angle
    def detect_right_left(self):
        position = "right" if self.landmarks_list[11][3] > self.landmarks_list[12][3] else "left"
        return position
def main():  
    previous_time = 0
    cap = cv2.VideoCapture("videos/4.mp4")
    detector = PoseEstimation()
    while True:
        success,img = cap.read()
        img = detector.findPoses(img)
        landmarkList = detector.findPosition(img)
        print(landmarkList)
        current_time = time.time()
        fbs = 1 /(current_time - previous_time)
        previous_time = current_time
        cv2.putText(img , str(int(fbs)) , (70,50) , cv2.FONT_HERSHEY_COMPLEX ,3 , (2550 ,0 ,255) ,3)
        cv2.imshow("Image", img)
        cv2.waitKey(1)
    


if __name__ == "__main__":
    main()