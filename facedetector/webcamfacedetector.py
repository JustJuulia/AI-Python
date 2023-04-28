import cv2
from random import randrange
facedata = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
# classifier -> detector
webcam = cv2.VideoCapture(0)
# captures video from webcam, the 0 stands for the default webcam, if you change it to file track of a video it will be checking video
while True:
    succesfulframeread, frame = webcam.read()
    #checks active frame
    grayframe = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # makes frame on grayscale bc it better works to capture it
    facecord = facedata.detectMultiScale(grayframe)
    # detects face, small/big and more than one or only one
    for (x, y, w, h) in facecord:
        # bercause facecord is a list, this loop will let us go through each detected face
        #sets x, y, w, h from facecord
        cv2.rectangle(frame, (x, y), (x + w, y + h), (randrange(256), randrange(256),  randrange(256)), 2)
        #draws rectangle up left cord(x + y), lower right vord(x + w, y + h); ( randrange(256), ...) refers to color; 3 to thickness,
        #rememebr that the color is bgr that means that the frist attribute is standing for GREEN not red\
        #where w = width, and h = height
    cv2.imshow('Detector of faces :0', frame)
    # what we will see in the popped up window. title and the frame
    key = cv2.waitKey(1)
    if key==113 or key==81:
        break
    #if Q pressed stop program, w ASCII 113 i 81 odpowiada Q i q
print("code done")





