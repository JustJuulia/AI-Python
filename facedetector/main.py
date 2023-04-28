import cv2
from random import randrange
facedata = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
# classifier -> detector
img = cv2.imread('face.jpg')
# basically you take a file trac,k and read it through; IMPORTING IMG INTO OPENCV
grayimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#makes img on grayscale bc it better works to capture it
facecord = facedata.detectMultiScale(grayimg)
#detects face, small/big and more than one or only one
print(facecord)
# shows coordinates of rectangle that will be around the face (left up corner, bottom right corner cords)
for (x, y, w, h) in facecord:
    # bercause facecord is a list, this loop will let us go through each detected face
    #sets x, y, w, h from facecord
    cv2.rectangle(img, (x, y), (x + w, y + h), (randrange(256), randrange(256),  randrange(256)), 2)
#draws rectangle up left cord(x + y), lower right vord(x + w, y + h); ( randrange(256), ...) refers to color; 3 to thickness,
#rememebr that the color is bgr that means that the frist attribute is standing for GREEN not red\
#where w = width, and h = height
cv2.imshow('Detector of faces :0', img)
# what we will see in the popped up window. title and the image
cv2.waitKey()
# the window will not shut down on itself, only when you press any key/ click the 'X'
print("code done")





