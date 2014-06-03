import cv2
import numpy as np
import matplotlib.pyplot as plt

samples = np.loadtxt('generalsamples.data',np.float32)
responses = np.loadtxt('generalresponses.data',np.float32)
responses = responses.reshape((responses.size,1))

model = cv2.KNearest()
model.train(samples,responses)

im = cv2.imread('test5.png')
out = np.zeros(im.shape,np.uint8)
gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray,(5,5),0)
#thresh = cv2.adaptiveThreshold(gray,255,1,1,11,2)
thresh = cv2.adaptiveThreshold(blur,255,1,1,11,2)

contours,hierarchy = cv2.findContours(thresh,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
mapaResultado = [0,0,0,0,0,0,0,0,0,0]
for cnt in contours:
    if cv2.contourArea(cnt)>50:
        [x,y,w,h] = cv2.boundingRect(cnt)
        if  h>23:
            cv2.rectangle(im,(x,y),(x+w,y+h),(0,255,0),1)
            #plt.imshow(im)
            #plt.show()
            roi = thresh[y:y+h,x:x+w]
            roismall = cv2.resize(roi,(10,10))
            #plt.imshow(roismall)
            #plt.show()
            roismall = roismall.reshape((1,100))
            roismall = np.float32(roismall)
            retval, results, neigh_resp, dists = model.find_nearest(roismall, k = 1)
            #print "RETVAL " + str(retval)
            #print "results " 
            #print  results
            #print "neigh_resp"
            #print neigh_resp
            #print "dist "
            #print dists
            string = str(int((results[0][0])))
            #print "Detectado = " + string
            mapaResultado[int((results[0][0]))] += 1
            cv2.putText(out,string,(x,y+h),0,1,(0,255,0))
plt.imshow(im)
#plt.imshow(out)
plt.show()
print "Cantidad de Elementos en el arreglo \n"
for i in xrange(0,10):
    print "Cantidad de %d es %d " % (i,mapaResultado[i])
#cv2.waitKey(0)