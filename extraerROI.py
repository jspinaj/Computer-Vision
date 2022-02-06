# -*- coding: utf-8 -*-
"""
Created on Sun Jan 16 18:12:48 2022

@author: j2seb
"""

#importación de librerías
import math 
import cv2     #Librería OpenCV
from IPython.display import Image   #Libreria impresión de imagenes
import numpy as np                  #Importación numpy
from matplotlib import pyplot as plt  #Impresión de gráficas
from PIL import Image


def extraccion_ROI(ruta,img_category):
    imagen=cv2.imread(ruta)
    imagen=cv2.cvtColor(imagen,cv2.COLOR_BGR2RGB)
    # copy to work over
    ima=imagen

    # tresholding
    ima_umb=umbral_hsv_rojo(ima)

    # morphologic transformations
    ima_llenado=llenado(ima_umb)
    ima_erosion=erosion(ima_llenado,18,18,10,"elipse")
    ima_apertura=apertura(ima_erosion,22,22,"elipse")
    ima_dilatacion=dilatacion(ima_apertura,20,20,10,"elipse")
    ima_final=cv2.bitwise_and(ima,ima,mask=ima_dilatacion)
    
    #Segementation
    imagenROI,contornos=deteccion_ROI(ima, ima_dilatacion)
    
    # feature detection
    imagenROI,ima_umbralizada,caracteristicas=obtener_caracteristicas(contornos,ima_final,ruta, img_category,imagenROI)
    
    # process
    ima_process=[ima,ima_umb,ima_llenado,ima_erosion,ima_apertura,ima_dilatacion,ima_final]

    return imagenROI,ima_umbralizada,caracteristicas,ima_process
    
    


def umbral_hsv_rojo(imagen):
    redBajo1 = np.array([0,85,10],np.uint8) 
    redAlto1 = np.array([8,255,255],np.uint8)
    redBajo2 = np.array([165,85,10],np.uint8) 
    redAlto2 = np.array([180,255,255],np.uint8) 
    img_hsv = cv2.cvtColor(imagen,cv2.COLOR_RGB2HSV)
    maskRed1 = cv2.inRange(img_hsv, redBajo1, redAlto1)
    maskRed2 = cv2.inRange(img_hsv, redBajo2, redAlto2)
    maskRed = cv2.add(maskRed1,maskRed2)
    return maskRed

def encontrar_semilla(imagen):
    semilla=None
    h,w=imagen.shape
    h=h-1
    w=w-1
    if imagen[0,0]==0:
        semilla=(0,0)
    elif imagen[h,0]==0:
        semilla=(h,0)
    elif imagen[h,w]==0:
        semilla=(h,w)
    elif imagen[0,w]==0:
        semilla=(0,w)
    else:
        print("no pudo encontrar una buena semilla")
    return semilla

def llenado(imagen):
    
    im_floodfill = imagen.copy()
    h, w = imagen.shape[:2]
    mask = np.zeros((h+2, w+2), np.uint8)
    seed=encontrar_semilla(imagen)

    # Floodfill from point (0, 0)
    cv2.floodFill(im_floodfill, mask, seed, 255);

     # Invert floodfilled image

    im_floodfill_inv = cv2.bitwise_not(im_floodfill)

    # Combine the two images to get the foreground.

    im_out = imagen | im_floodfill_inv
    return im_out

def erosion(imagen,a,b,n,tipo):
    if tipo=="rect":
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(a,b))
    elif tipo=="elipse":
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(a,b))
    elif tipo=="cruz":
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(a,b))
    else:
        print("tipo de elemento estructurante invalido")
        return imagen
    ima=imagen.copy()
    for i in range(1,n):
        ima= cv2.erode(ima,kernel)
    return ima

def dilatacion(imagen,a,b,n,tipo):
    if tipo=="rect":
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(a,b))
    elif tipo=="elipse":
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(a,b))
    elif tipo=="cruz":
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(a,b))
    else:
        print("tipo de elemento estructurante invalido")
        return imagen
    ima=imagen.copy()
    for i in range(1,n):
        ima= cv2.dilate(ima,kernel,n)
    return ima

def apertura(imagen,a,b,tipo):
    if tipo=="rect":
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(a,b))
    elif tipo=="elipse":
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(a,b))
    elif tipo=="cruz":
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(a,b))
    else:
        print("tipo de elemento estructurante invalido")
        return imagen
    ima= cv2.morphologyEx(imagen, cv2.MORPH_OPEN, kernel)
    return ima

def deteccion_ROI(imagen_inicial,imagen_regiones):
    ima_contornos=imagen_inicial.copy()
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
    edges = cv2.morphologyEx(imagen_regiones, cv2.MORPH_GRADIENT, kernel)
    contours, hierarchy = cv2.findContours(edges , cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
      
    return ima_contornos,contours

def obtener_caracteristicas(contornos,imagen,img_filename, img_category,imagenROI): 
    caracteristicas=[]
    imagenes=[]
    imagenes_umbralizadas=[]
    for i in range(len(contornos)):
        cont=contornos[i]
        #Area del contorno
        area=cv2.contourArea(cont)
        #Longitud del contorno
        long_arc=cv2.arcLength(cont,True)
        #Posición del centroide
        M = cv2.moments(cont)
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])
        cent=(cx,cy)
        #compacidad
        comp=long_arc**2/area
        #redondez
        redon=4*np.pi*area/long_arc**2
        #mínimo rectangulo
        rect = cv2.minAreaRect(cont)
        box = cv2.boxPoints(rect)

        #Longitudes rectangulo
        a=cv2.arcLength(np.array([box[0],box[1]]),False)
        b=cv2.arcLength(np.array([box[0],box[3]]),False)

        largo=np.max([a,b])
        corto=np.min([a,b])
        rel=largo/corto
        
        #rectangulo limite
        x,y,w,h=cv2.boundingRect(cont)
        rect_lim={"x":x,
                  "y":y,
                  "w":w,
                  "h":h}
        prop=[img_filename,"defecto"]+[0]*9+[i] 
        #cortar la imagen
        imagen_cortada=imagen[y:y+h,x:x+w,:]
        #hallar los momentos de Hu
        ima_gris=cv2.cvtColor(imagen_cortada,cv2.COLOR_BGR2GRAY)
        ret,th1 = cv2.threshold(ima_gris,100,255,cv2.THRESH_BINARY)
        moments = cv2.moments(th1)
        huMoments = cv2.HuMoments(moments).flatten()
        #print(huMoments)
        

        for j in range(0,7):
            if huMoments[j]!=0:
                huMoments[j] =  -1* np.sign( huMoments[j]) * np.log10(abs(huMoments[j]))
                
        huMoments=huMoments.tolist()
        
        prop=[img_filename, img_category,long_arc,redon,rel]+huMoments+[i]
        
        if rel<1.5 and redon>0.7:
            cv2.rectangle(imagenROI,(x,y),(x+w,y+h),(120,255,0),15)
            cv2.drawContours(imagenROI, contornos, i, (0, 0, 255), 20)
            cv2.putText(imagenROI,str(i),(x,y-10),2,10,(0,255,0),10)       
            #caracteristicas.append(diccionario)
            #imagenes.append(imagen_cortada)
            imagenes_umbralizadas.append(th1)
        else:
            prop[1]="ruido"    
            cv2.drawContours (imagenROI, contornos, i, (255, 0, 0), 20)
            cv2.putText(imagenROI,str("r"+str(i)),(x+30,y+10),2,10,(0,255,0),10)
        
        caracteristicas.append(prop)

    return imagenROI, imagenes_umbralizadas, caracteristicas

    # Functions

