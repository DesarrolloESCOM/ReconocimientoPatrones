import cv2
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import sys

def contadorLetras(cadena):
    mapaResultado = [0,0,0,0,0,0,0,0,0,0]
    for i in xrange(0,len(cadena)):
        mapaResultado[int(cadena[i])] += 1
    return mapaResultado

samples = np.loadtxt('generalsamples.data',np.float32)
responses = np.loadtxt('generalresponses.data',np.float32)
responses = responses.reshape((responses.size,1))

model = cv2.KNearest()
model.train(samples,responses)
#se cargan los argumentos
argumentos = sys.argv
print len(argumentos)
if len(argumentos)<=1:
    argumentos = [argumentos[0],'','']
if len(argumentos[1])>0:
    imagenPrueba = argumentos[1]
else:
    imagenPrueba = './Imagenes/PruebaMini.png'
im = cv2.imread(imagenPrueba)
out = np.zeros(im.shape,np.uint8)
gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray,(5,5),0)
#thresh = cv2.adaptiveThreshold(gray,255,1,1,11,2)
thresh = cv2.adaptiveThreshold(blur,255,1,1,11,2)

contours,hierarchy = cv2.findContours(thresh,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
mapaResultado = [0,0,0,0,0,0,0,0,0,0]

#plt.ion()
objetoImagen = None
contadorAuxiliar = 0

for cnt in contours:
    if cv2.contourArea(cnt)>50:
        [x,y,w,h] = cv2.boundingRect(cnt)
        if  h>22:
            cv2.rectangle(im,(x,y),(x+w,y+h),(0,255,0),1)
            roi = thresh[y:y+h,x:x+w]
            """
            if contadorAuxiliar is 0:
                objetoImagen = plt.imshow(roi)
                plt.show()
            else:
                objetoImagen.set_data(roi)
            plt.pause(0.0001)
            contadorAuxiliar+=1

            raw_input("Continuar ... ")"""
            roismall = cv2.resize(roi,(10,10))
            roismall = roismall.reshape((1,100))
            roismall = np.float32(roismall)
            retval, results, neigh_resp, dists = model.find_nearest(roismall, k = 1)
            string = str(int((results[0][0])))
            mapaResultado[int((results[0][0]))] += 1
            cv2.putText(out,string,(x,y+h),0,1,(0,255,0))
#plt.imshow(im)
#plt.show()
#print "Cantidad de Elementos en el arreglo \n"
#for i in xrange(0,10):
#    print "Cantidad de %d es %d " % (i,mapaResultado[i])
#fig,ax = plt.figure()
numeros = ('0','1','2','3','4','5','6','7','8','9')
y_pos = np.arange(len(numeros))

if len(argumentos[2]) > 0:
    cantidadOriginal = contadorLetras(argumentos[2])
    error = []
    for index in xrange(0,10):
        valor = float(cantidadOriginal[index]-mapaResultado[index])/cantidadOriginal[index]
        error.append(valor) 
else:
    error = [0.1] * 10
plt.barh(y_pos, mapaResultado, xerr=error, align='center', alpha=0.4)
plt.yticks(y_pos,numeros)
plt.xlabel('Rendimiento')
plt.show()