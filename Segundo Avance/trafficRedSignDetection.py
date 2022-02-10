# -*- coding: utf-8 -*-
"""
Created on Sun Jan 16 18:12:48 2022

@author: miarodriguezur, jspinaj
"""
#importación de librerías
import cv2     #Librería OpenCV
from IPython.display import Image   #Libreria impresión de imagenes
import numpy as np                  #Importación numpy
from matplotlib import pyplot as plt  #Impresión de gráficas
from PIL import Image


#--------------------------------------Main Function----------------------------------------

def detectROI(path):
    #Read the Image
    image=cv2.imread(path)
    image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    
    ## Detect ROI on Cr Channel##
    # thresholding image
    ima_umb=redTh(image)
    
    #filling image
    ima_fill=filling(ima_umb)
    
    #morphological operations
    ima_ero=erode(ima_fill,18,18,9,"elipse")
    ima_open=opening(ima_ero,22,22,"elipse")
    ima_dil_Cr=dilate(ima_open,20,20,9,"elipse")
    
    # Segmentation
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
    edges_Cr = cv2.morphologyEx(ima_dil_Cr, cv2.MORPH_GRADIENT, kernel)
    contours_Cr, _ = cv2.findContours(edges_Cr , cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    #Fetch contour characteristics
    characteristics_Cr=getContourCharacteristics(contours_Cr)
    
    ##Filtered out##
    #Filtering Cr characteristics
    delete=[]
    for i in range(len(characteristics_Cr)):
        rel=characteristics_Cr[i].get("Relacion entre lados")
        if rel>1.5 :
            delete.append(characteristics_Cr[i])
    for char in delete:
        characteristics_Cr.remove(char)      
    
    #Fetch Region characteristics
    char_Cr,region_Cr=getRegionCharacteristics(characteristics_Cr,ima_dil_Cr,image)
    
    characteristics=char_Cr
    region=region_Cr
    
    ##Show ROI on the image
    ima_ROI=image.copy()
    cv2.drawContours(ima_ROI, contours_Cr, -1, (255, 0, 0), 20)
    for i in range(len(characteristics)):
        
        x=characteristics[i].get("x")
        y=characteristics[i].get("y")
        w=characteristics[i].get("w")
        h=characteristics[i].get("h")

        cv2.rectangle(ima_ROI,(x,y),(x+w,y+h),(120,255,0),15)
        cv2.putText(ima_ROI,str(i),(x,y-10),2,10,(0,255,0),10)
    
    return characteristics, region, ima_ROI


def dict2list(dictionary,keys=None):
    values=[]
    if keys==None:
        keys=dictionary.keys()
    for key in keys:
        values.append(dictionary.get(key))
    return keys, values
    
    
#--------------------------------- Tresholding Function------------------------------------   

def blackTh(image):
    sz=201
    c=30
    ima=cv2.cvtColor(image.copy(),cv2.COLOR_RGB2GRAY)
    th = cv2.adaptiveThreshold(ima,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY_INV,sz,c)
    return th

def redTh(image):
    sz=201
    c=-2
    ima=cv2.cvtColor(image.copy(),cv2.COLOR_RGB2YCrCb)
    Y,Cr,Cb=cv2.split(ima)
    #Threshold Cr Channel
    thCr = cv2.adaptiveThreshold(Cr,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,sz,c)
    return thCr

#--------------------------------- Filling closed contours------------------------------------ 

def filling(image):
    
    im_floodfill = image.copy()
    h, w = image.shape
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

    im_out = image | im_floodfill_inv
    return im_out

#--------------------------------- Morphological Operations------------------------------------ 

def dilate(imagen,a,b,n,tipo):
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

def erode(imagen,a,b,n,tipo):
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

def opening(imagen,a,b,tipo):
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

#--------------------------- Fetch characterictics Function-----------------------------   

def getContourCharacteristics(contours):
    characteristics=[]
    for cont in contours:
        #Area del contorno
        area=cv2.contourArea(cont)
        #Longitud del contorno
        long=cv2.arcLength(cont,True)
        #compacity
        comp=long**2/area
        #roundness
        redon=4*np.pi*area/long**2
        #min rectangle
        rect = cv2.minAreaRect(cont)
        box = cv2.boxPoints(rect)
        #Min Rectangle Relation
        a=cv2.arcLength(np.array([box[0],box[1]]),False)
        b=cv2.arcLength(np.array([box[0],box[3]]),False)
        longer=np.max([a,b])
        shorter=np.min([a,b])
        rel=longer/shorter
        #bounding rect
        x,y,w,h=cv2.boundingRect(cont)
        dictionary={"Area":area,
                    "Perimetro":long,
                    "Compacidad":comp,
                    "Redondez":redon,
                    "Minimo Rectangulo":box,
                    "Relacion entre lados":rel,
                    "x":x,
                    "y":y,
                    "w":w,
                    "h":h,
                    "Contorno":cont}
        characteristics.append(dictionary)
    return characteristics
        
def getRegionCharacteristics(characteristics,mask,image):
    image_gray=cv2.cvtColor(image.copy(),cv2.COLOR_RGB2GRAY)
    image_masked=cv2.bitwise_and(image_gray, mask, mask = None)
    region_char=[]
    region_img=[]
    for i in range(len(characteristics)):
        x_ini=characteristics[i].get("x")
        x_fin=characteristics[i].get("x")+characteristics[i].get("w")
        y_ini=characteristics[i].get("y")
        y_fin=characteristics[i].get("y")+characteristics[i].get("h")
        w=characteristics[i].get("w")
        h=characteristics[i].get("h")
        cont=characteristics[i].get("Contorno")
        #cropping image and mask
        cropped_image=image_masked[y_ini:y_fin,x_ini:x_fin]
        cropped_mask=mask[y_ini:y_fin,x_ini:x_fin]
        #histogram
        hist = cv2.calcHist([cropped_image], [0], cropped_mask, [256], [0, 255]) 
        hist = hist/np.sum(hist)
        r=otsu(hist)
        _,th = cv2.threshold(cropped_image,r,255,cv2.THRESH_BINARY)
        #Move contour to cropped image coordinates
        cont[:,:,1]=cont[:,:,1]-y_ini
        cont[:,:,0]=cont[:,:,0]-x_ini
        #Eliminate external regions
        th1=edgeExternalRemoval(th.copy(),cont,w,h)
        #calculate areas of region, threshold and threshold with edge removal
        filled_area=np.sum(th!=0)
        filled_area1=np.sum(th1!=0)
        total_area=np.sum(cropped_mask!=0)
        #update threshold if edge region removal is valid
        area_percentage_eliminated=(filled_area-filled_area1)/total_area
        if area_percentage_eliminated<0.2 :
            th=th1
            filled_area=filled_area1
        #Calculate Characteristics
        moments = cv2.moments(th)
        huMoments = cv2.HuMoments(moments)
        huMoments = -1* np.sign(huMoments) * np.log10(np.abs(huMoments))
        fill_percentage=filled_area*100/total_area
        H=entropy(hist)
        #add region characteristics to dictionary
        diccionario={"Primer Momento":huMoments[0,0],
                     "Segundo Momento":huMoments[1,0],
                     "Tercer Momento":huMoments[2,0],
                     "Cuarto Momento":huMoments[3,0],
                     "Quinto Momento":huMoments[4,0],
                     "Sexto Momento":huMoments[5,0],
                     "Septimo Momento":huMoments[6,0],
                    "Entropia":H,
                    "Porcentaje de Area Rellena":fill_percentage,
                    "Umbral usado":r}
        characteristics[i].update(diccionario)
        #remove contour from dictionary
        characteristics[i].pop("Contorno")
        region_img.append(th)
    return characteristics,region_img
        
        
        
#--------------------------- Otsu's Threshold Algoritm-----------------------------

def otsu(hist):
    Q=np.zeros(256)
    a=np.zeros(256)
    b=np.zeros(256)
    w0=np.zeros(256)
    w1=np.zeros(256)
    u0=np.zeros(256)
    u1=np.zeros(256)
    a[0]=hist[0]
    b[0]=hist[0]
    for i in range(255):
        a[i+1]=a[i]+hist[i+1]
        b[i+1]=b[i]+(i+2)*hist[i+1]
    for i in range(255):
        w0[i]=a[i]
        w1[i]=a[255]-a[i]
        if w0[i]!=0:
            u0[i]=b[i]/w0[i]
        if w1[i]!=0:
            u1[i]=(b[255]-b[i])/w1[i]
    Q=w0*w1*(u1-u0)
    r=np.argmax(Q)
    return r

#--------------------------- Entropy Function-----------------------------

def entropy(hist):
    a=np.where(hist==0,0.1,hist)
    H=np.sum(hist*np.log2(a))
    return H

#----------------------Eliminating edge noise-------------------

def edgeExternalRemoval(thresh, contour,w,h):
        #remove bounding regions
        for point in contour:
            x=point[0][0]
            y=point[0][1]
            sx=int(np.sign(w/2-x))
            sy=int(np.sign(h/2-y))
            desp=(x+1*sx,y+1*sy)
            cv2.floodFill(thresh, None, desp, 0);
        return thresh
