
import cv2
import numpy as np
import time

rec = cv2.face.LBPHFaceRecognizer_create()
rec.read(�trainer//trainer.xml�)
det = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

font = cv2.FONT_HERSHEY_SIMPLEX
kameraAc = cv2.VideoCapture(0)

while True:

ret,cerceve = kameraAc.read() #kameradan gelen veriyi iki de�i�kene atad�k
gri = cv2.cvtColor(cerceve,cv2.COLOR_BGR2GRAY)
yuzler = det.detectMultiScale(gri,1.3,5) #grideki resmi al 1.3 ile �l�ekle 5�er de yanlardan
for (x,y,c,d) in yuzler: #kareyi olu�turmak i�in
cv2.rectangle(cerceve,(x,y),(x+c,y+d),(255,0,0),2) #dikd�rtgene �er�eveyi at�x-y al�mavi yap�
Id,conf = rec.predict(gri[y:y+d,x:x+c])
if conf<95:
if Id == 5:
Id = "Doga"

else:
Id="Kisi taninmiyor!!!"
else:
Id = "Kisi taninmiyor!!!"
cv2.putText(cerceve, str(Id), (x,y), font, 0.75, (255,255,255), 2, cv2.LINE_AA)
cv2.imshow("YuzTanima", "cerceve")
key = cv2.waitKey(1) & 0xFF
if key == ord('q'):
break
