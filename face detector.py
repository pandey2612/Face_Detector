"""
Created on Sat Feb 16 00:15:31 2019

@author: Himanshu pandey

"""

#importing Library Computer vision
import cv2    

#making variable cap to capture images from the webcam

cap = cv2.VideoCapture(0)

#creating a variable to cascade the values from the XML file
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

#As we are dealing with the frames so we need to make the loop to make a video
while True:
    
    #ret will get bool value True or False and frame will get numpy values of the colour of the frame
    
    ret, frame = cap.read()

    #converting colourful image to a grayscale not necessary but for the Best Detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    #counting the faces present in the frames through face_cascade objects
    faces = face_cascade.detectMultiScale(gray, 1.2, 5)


    #xand y are the coordinates of the first point where the face detected
    #W is the width of the face
    #h is the height of the face
    #these values are sufficient for detecting face
    
    
    #as we have more number if faces so we are running loop
    for (x,y,w,h) in faces:
         cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
        
        
     # resizing image for better detection   
    frame=cv2.resize(frame,(1366,768))    


    #showing the frame to the user
    cv2.imshow('frame',frame)
    
    #taking input from keyboard to exit the loop
    key=cv2.waitKey(1)
    if key== ord('q'):
        break


#exiting the webcam
cap.release()

#destroying the window
cv2.destroyAllWindows()