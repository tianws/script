#!/usr/local/bin/python3
import datetime
import numpy as np
import cv2

loc = "beijing_qianshi_"

now = datetime.datetime.now()
now = now.strftime("%Y-%m-%d-%H-%M-%S")


fileName = loc + now + ".avi"

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,720)
fps = cap.get(cv2.CAP_PROP_FPS)
w = cap.get(cv2.CAP_PROP_FRAME_WIDTH) 
h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)


# fourcc = cv2.VideoWriter_fourcc(*'XVID')
fourcc = cv2.cv.CV_FOURCC("D","I","B"," ")
out = cv2.VideoWriter(fileName ,fourcc,fps,(1280,720))

while(cap.isOpened()):
    ret, frame = cap.read()
    if ret==True:
        frame = cv2.flip(frame,1)
        frame = cv2.flip(frame,0)

        out.write(frame)
#        frame1 = cv2.resize(frame,(640,480))
#        flip(frame,frame,0)
#        flip(frame,frame,1)
        cv2.imshow('frame',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

print("fps:",fps,w,h)
cap.release()
out.release()
cv2.destroyAllWindows()
