import cv2
import numpy as np
from ultralytics import YOLO
import cvzone
import numpy as np

def RGB(event, x, y, flags, param):
    if event==cv2.EVENT_MOUSEMOVE:
        point=[x,y]
        print(point)
#cv2.namedWindow('RGB')
#cv2.setMouseCallback('RGB',RGB)
cap=cv2.VideoCapture('people1.avi')

cy1=286
cy2=350
inp={}
enter=[]
exit=[]
exp={}
count=0
offset=8
model=YOLO('yolo11s')
names=model.model.names
w,h,fps=(int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH,cv2.CAP_PROP_FRAME_HEIGHT,cv2.CAP_PROP_FPS))
video_writer= cv2.VideoWriter("countpeople.mp4",cv2.VideoWriter.fourcc(*"mp4v"),fps,(w,h))
    
#line1=(490,286),(700,286)
#line2=(840,300),(400,300)
while True:
    ret,frame=cap.read()
    if ret:
        
        frame=cv2.resize(frame,(1200,640))
        results=model.track(frame,persist=True,classes=0)
        if results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.int().cpu().tolist()  
            class_ids = results[0].boxes.cls.int().cpu().tolist()  
            track_ids = results[0].boxes.id.int().cpu().tolist()  
            confidences = results[0].boxes.conf.cpu().tolist()
            for box, class_id, track_id, conf in zip(boxes, class_ids, track_ids, confidences):
                c = names[class_id]
                x1, y1, x2, y2 = box
                cx=(x1+x2)//2
                cy=(y1+y2)//2
                
                if cy1<(cy+offset) and cy1>(cy-offset):
                    inp[track_id]=(cx,cy)
                if track_id in inp:
                    if cy2<(cy+offset) and cy2>(cy-offset):
                        cv2.circle(frame,(cx,cy),4,(255,0,0),-1)
                        cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
                        cvzone.putTextRect(frame,f'{track_id}',(x1,y2),1,1)
                        cvzone.putTextRect(frame,f'{c}',(x1,y1),1,1)
                        if enter.count(track_id)==0:
                           enter.append(track_id)
###################################################################################################
                           
                if cy2<(cy+offset) and cy2>(cy-offset):
                    exp[track_id]=(cx,cy)
                if track_id in exp:
                    if cy1<(cy+offset) and cy1>(cy-offset):
                        cv2.circle(frame,(cx,cy),4,(0,0,255),-1)
                        cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
                        cvzone.putTextRect(frame,f'{track_id}',(x1,y2),1,1)
                        cvzone.putTextRect(frame,f'{c}',(x1,y1),1,1)
                        if exit.count(track_id)==0:
                           exit.append(track_id)          
        cv2.line(frame,(400,286),(1200,286),(255,0,0),1)
        cv2.line(frame,(400,350),(1200,350),(0,0,255),1)
        ##cv2.line(frame,(400,1400),(1700,1400),(0,0,255),1)
        enterp=len(enter)
        cvzone.putTextRect(frame,f'ENTER_PERSON_COUNT:{enterp}',(50,60),2,2)
        exitp=len(exit)
        cvzone.putTextRect(frame,f'EXIT_PERSON_COUNT:{exitp}',(50,160),2,2)
        print(inp)
        print(len(enter))
        #cv2.imshow("FRME",frame)
        #cv2.namedWindow('FRME')
        #cv2.setMouseCallback('FRME',RGB)
         
    cv2.imshow("FRME",frame)
    #video_writer.write(frame)  
    if cv2.waitKey(1) & 0xFF==ord('q'):
        break
     

cap.release()
cv2.destroyAllWindows()