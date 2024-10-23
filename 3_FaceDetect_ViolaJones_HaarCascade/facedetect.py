#Imports
import numpy as np
import cv2

#On charge les classifieurs préexistants pour le visage et les yeux
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier("haarcascade_eye.xml")

#Charge la photo sur laquelle on travaille
img = cv2.imread("photo.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray, 1.3, 5)

#Pour chaque visage détecté on dessine un rectangle
for (x,y,w,h) in faces:
    img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = img[y:y+h, x:x+w]
    eyes = eye_cascade.detectMultiScale(roi_gray)
    # Pour chaque oeil détecté on dessine un rectangle
    for (ex,ey,ew,eh) in eyes:
        cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
        cv2.imshow('img',img)
cv2.waitKey(0) #On attend la pression d'une touche
cv2.destroyAllWindows() #libère les ressources liées à la fenêtre