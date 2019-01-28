import numpy as np
import cv2
from PIL import Image

rec = cv2.face.LBPHFaceRecognizer_create()
det = cv2.CascadeClassifier(haarcascade_frontalface_default.xml)

def  resimleriCek(path):
resimler=[os.path.join(path,f) for f in os.listdir(path)]
yuzOrnekler=[]
idler=[]
for resim in resimler:
PIL_resim = Image.open(resim).convert(�L�)
numpy_resim = np.array(PIL_resim,�uint8?)
id_resim = int(os.path.split(resim)[-1].split(�.�)[1])
print(id_resim)
y�zler = det.detectMultiScale(numpy_resim)
for(x,y,c,d) in y�zler:
yuzOrnekler.append(numpy_resim[y:y+d,x:x+c])
idler.append(id_resim)
return yuzOrnekler,idler

y�zler, idler = resimleriCek(�dataset//�)
rec.train(y�zler, np.array(idler))
rec.save(�trainer//trainer.xml�)
