import cv2

face_detector=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
smile_detector=cv2.CascadeClassifier('haarcascade_smile.xml')
eye_detector=cv2.CascadeClassifier('haarcascade_eye.xml')
webcam=cv2.VideoCapture(0)
while True:
    success, frame=webcam.read()
    frame_gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces=face_detector.detectMultiScale(frame_gray)
    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y), (x+w,y+h), (100,200,50), 4)
        face=frame[y:y+h,x:x+w]
        face_gray=cv2.cvtColor(face,cv2.COLOR_BGR2GRAY)
        smiles=smile_detector.detectMultiScale(face_gray,scaleFactor=1.7,minNeighbors=25)
        for (X,Y,W,H) in smiles:
            cv2.rectangle(face,(X,Y), (X+W,Y+H), (50,100,200), 4)
            cv2.putText(frame,'Smiling',(x,y+h+30),fontScale=2,fontFace = cv2.FONT_HERSHEY_SIMPLEX,color=(255,255,255))
        eyes=eye_detector.detectMultiScale(face_gray,scaleFactor=1.7,minNeighbors=25)
        for (X,Y,W,H) in eyes:
            cv2.rectangle(face,(X,Y), (X+W,Y+H), (50,100,100), 5)
    cv2.imshow("Smile",frame)
    
    key=cv2.waitKey(1)
    if key==13:
        break
webcam.release()
cv2.destroyAllWindows()

import cv2
print(cv2.__file__)