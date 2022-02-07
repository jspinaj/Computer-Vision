# -*- coding: utf-8 -*-
"""
Created on Sun Jan 16 18:12:48 2022

@author: miarodriguezur
"""
#importación de librerías
import cv2     #Librería OpenCV
from IPython.display import Image   #Libreria impresión de imagenes
import numpy as np                  #Importación numpy
from matplotlib import pyplot as plt  #Impresión de gráficas
from PIL import Image


#--------------------------------------Main Function----------------------------------------

def senalRectangular(ruta):
    imagen=cv2.imread(ruta)
    imagen=cv2.cvtColor(imagen,cv2.COLOR_BGR2RGB)

    # copy to work over
    ima=imagen
    
    # tresholding
    ima_umb=negro_adapt(ima,201,30)
    
    # filling and morphological operations
    ima_llenado=llenado(ima_umb)
    ima_erosion=erosion(ima_llenado,20,5,8,"rect")
    ima_apertura=apertura(ima_erosion,15,15,"rect")
    ima_dilatacion=dilatacion(ima_apertura,20,6,10,"rect")
    ima_final=cv2.bitwise_and(ima,ima,mask=ima_dilatacion)
    
    # Countours
    ima_contornos=ima.copy()
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
    edges = cv2.morphologyEx(ima_dilatacion, cv2.MORPH_GRADIENT, kernel)
    contours, hierarchy = cv2.findContours(edges , cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    caracteristicas, imagenesFinales =getCharacteristics(contours, ima_final)

    
#--------------------------------- Tresholding Function------------------------------------   

def negro_adapt(imagen,sz,c):
    ima=cv2.cvtColor(imagen.copy(),cv2.COLOR_RGB2GRAY)
    th = cv2.adaptiveThreshold(ima,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY_INV,sz,c)
    return th

#----------------- Morphologic transformation and filling Functions------------------------- 

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
    for i in range(n):
        ima= cv2.dilate(ima,kernel,n)
    return ima

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
    for i in range(n):
        ima= cv2.erode(ima,kernel)
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

def llenado(imagen):
    
    im_floodfill = imagen.copy()
    h, w = imagen.shape
    h=h-1
    w=w-1
    #mask = np.zeros((h+2, w+2), np.uint8)
    seed=[(0,0),(0,h),(w,h),(w,0)]

    
    # Floodfill from point (0, 0)
    cv2.floodFill(im_floodfill, None, seed[0], 255);
    cv2.floodFill(im_floodfill, None, seed[1], 255);
    cv2.floodFill(im_floodfill, None, seed[2], 255);
    cv2.floodFill(im_floodfill, None, seed[3], 255);

     # Invert floodfilled image

    im_floodfill_inv = cv2.bitwise_not(im_floodfill)

    # Combine the two images to get the foreground.

    im_out = imagen | im_floodfill_inv
    return im_out
    
#--------------------------- Fetch characterictics Function-----------------------------   
    
def getCharacteristics(contornos,imagen):
    caracteristicas=[]
    caracteristicasFinales=[]
    imagenes=[]
    for i in range(len(contornos)):
        cont=contours[i]
        #Area del contorno
        area=cv2.contourArea(cont)
        #Longitud del contorno
        long=cv2.arcLength(cont,True)
        #Posición del centroide
        M = cv2.moments(cont)
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])
        cent=(cx,cy)
        #compacidad
        comp=long**2/area
        #redondez
        redon=4*np.pi*area/long**2
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
        
        #cortar la imagen
        imagen_cortada=imagen[y:y+h,x:x+w,:]
        #hallar los momentos de Hu
        ima_gris=cv2.cvtColor(imagen_cortada,cv2.COLOR_BGR2GRAY)
        ret,th1 = cv2.threshold(ima_gris,100,255,cv2.THRESH_BINARY)
        moments = cv2.moments(th1)
        huMoments = cv2.HuMoments(moments).flatten()    
        
        for j in range(0,7):
            if huMoments[j]!=0:
                huMoments[j] =  -1* np.sign( huMoments[j]) * np.log10(abs(huMoments[j]))
                
        huMoments=huMoments.tolist()
        
        diccionario={"Area":area,
                    "Perimetro":long,
                    "Centroide":cent,
                    "Compacidad":comp,
                    "Redondez":redon,
                    "Minimo Rectangulo":box,
                    "Lado largo":largo,
                    "Lado corto":corto,
                    "Relacion entre lados":rel,
                    "Rectangulo limite":rect_lim,
                    "Momentos de Hu":huMoments}
        
        caracteristicas.append(diccionario)
        
        #obtiene la relación de lados
        com=caracteristicas[i].get("Compacidad")
        if com<24 and com>21 :
            rect_lim=caracteristicas[i].get("Rectangulo limite")
            x_ini=rect_lim.get("x")
            x_fin=rect_lim.get("x")+rect_lim.get("w")
            y_ini=rect_lim.get("y")
            y_fin=rect_lim.get("y")+rect_lim.get("h")
            imagen_cortada=ima_final[y_ini:y_fin,x_ini:x_fin,:]
            caracteristicasFinales.append(caracteristicas[i])
            imagenes.append(imagen_cortada)
        
    return caracteristicasFinales, imagenes
