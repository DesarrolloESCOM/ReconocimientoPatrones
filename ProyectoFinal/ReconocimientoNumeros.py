import sys

import numpy as np
import cv2
import matplotlib.pyplot as plt
im = cv2.imread('Entrenamiento2.png')
#print "la lees??"
im3 = im.copy()

gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray,(5,5),0)
thresh = cv2.adaptiveThreshold(blur,255,1,1,11,2)


contours,hierarchy = cv2.findContours(thresh,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)

samples =  np.empty((0,100))
responses = []
keys = [i for i in range(48,58)]
contadorAuxiliar = 0

fig=plt.figure()

plt.ion()
objetoImagen = None

contadorAuxiliar = 0
for cnt in contours:
    if cv2.contourArea(cnt)>50:
        [x,y,w,h] = cv2.boundingRect(cnt)
        if  h>23:
            cv2.rectangle(im,(x,y),(x+w,y+h),(0,0,255),2)
            roi = thresh[y:y+h,x:x+w]
            if contadorAuxiliar is 0:
                objetoImagen = plt.imshow(roi)
                plt.show()
            else:
                objetoImagen.set_data(roi)            
            plt.pause(0.0001)
            contadorAuxiliar += 1
            roismall = cv2.resize(roi,(10,10))
            
            print "Que numero es? "
            valor = int(raw_input())
            responses.append(valor)
            sample = roismall.reshape((1,100))
            #plt.imshow(samples,cmap='gray')
            #plt.show()
            samples = np.append(samples,sample,0)
plt.close()
responses = np.array(responses,np.float32)
responses = responses.reshape((responses.size,1))
print "training complete"

np.savetxt('generalsamples.data',samples)
np.savetxt('generalresponses.data',responses)
