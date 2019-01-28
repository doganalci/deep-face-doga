import numpy as np
import cv2
import matplotlib.pyplot as plt
import time
from PIL import Image
face_id = int(input("Id gir\n"))
count = 0
yuz=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
kameraac=cv2.VideoCapture(0) #genelde cap=cv2... diye atama yap�l�r
while True:
    ret,cerceve=kameraac.read() #kameradan gelen veriyi iki de�i�kene atad�k
    gri=cv2.cvtColor(cerceve,1) #arka plana att��� g�rselleri renkli �ekilde atmas� i�in
    yuzler =yuz.detectMultiScale(gri,1.3,5) #grideki resmi al 1.3 ile �l�ekle 5'er de yanlardan
    for (x,y,c,d) in yuzler: #kareyi olu�turmak i�in
        cv2.rectangle(cerceve,(x,y),(x+c,y+d),(255,0,0),2) #dikd�rtgene �er�eveyi at--x-y al---mavi yap--
    #cv2.putText(�er�eve, 'Ahmet', (x,y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255,255,255), 2, cv2.LINE_AA)
        count += 1
        cv2.imwrite("/dataset//user." + str(face_id) + '.' + str(count) + ".jpg", gri[y:y+d,x:x+c])
        key = cv2.waitKey(1) & 0xFF
    if count > 50:
        break
