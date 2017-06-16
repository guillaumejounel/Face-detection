# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
from skimage import io
from skimage import util
from skimage import color
#from skimage import feature
from sklearn import svm
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from PIL import Image

absolutePath = "/Users/guillaume/Cloud/WORK/UTC/GI02/SY32/TDXu/Projet/"
pathTrain = absolutePath + "projetface/train/"
pathTest =  absolutePath + "projetface/test/"
pathFile = absolutePath + "projetface/label.txt"
data = np.loadtxt(pathFile)


def smallestFace(data):
    return int(np.min(data[:,3:]))

newSize = smallestFace(data)
print("smallestFace =", newSize)

def afficherImgRect(n, data, pathTrain):
    # Charger l'image
    img = np.array(Image.open(pathTrain +"%04d"%(n)+".jpg"), dtype=np.uint8)
    # Créer la figure et les axes
    fig,ax = plt.subplots(1)
    # Afficher l'image
    ax.imshow(img)
    # Créer le rectangle 
    xcorner, ycorner, width, height = data[n-1][1:]
    rect = patches.Rectangle((xcorner, ycorner), width, height,linewidth=2,edgecolor='r', facecolor='none')
    # Ajouter le rectangle sur l'image
    ax.add_patch(rect)
    plt.show()

afficherImgRect(14, data, pathTrain)

def cropImage(n, data, pathTrain):
    img = np.array(Image.open(pathTrain +"%04d"%(n)+".jpg"), dtype=np.uint8)
    x, y, w, h = map(int, data[n-1][1:])
    img = img[y:y+h, x:x+w]
    io.imshow(img)

cropImage(14,data,pathTrain)

from skimage.transform import resize
def cropImageSquare(n, data, pathTrain, newsize):
    # Chargement de l'image
    img = np.array(Image.open(pathTrain +"%04d"%(n)+".jpg"), dtype=np.uint8)
    # Récupération du rectangle
    x, y, w, h = map(int, data[n-1][1:])
    # Transformation en carré
    minwh = min(w,h)
    x+=(w-minwh)//2 # Pour centrer le carré //2
    y+=(h-minwh)//2
    # Recadrement de l'image
    img = img[y:y+minwh, x:x+minwh]
    # Redimensionnement de l'image
    return resize(img, (newsize, newsize), mode='reflect')

fig,ax = plt.subplots(1)
ax.imshow(cropImageSquare(14,data,pathTrain, newSize))
plt.show()

def exemplesPositifs(data, pathTrain, newsize):
    images = np.zeros((len(data),newsize,newsize))
    for i in range(len(data)):
        images[i] = color.rgb2gray(cropImageSquare(i+1, data, pathTrain, newsize))
    return images

exPos = exemplesPositifs(data, pathTrain, newSize)
io.imshow(exPos[13])

def recouvrement(x1, y1, w1, h1, x2, y2, w2, h2):
    xinter = max(0, min(x1+w1,x2+w2) - max(x1,x2))
    yinter = max(0, min(y1+h1,y2+h2) - max(y1,y2));
    ainter = xinter * yinter;
    aunion = (w1*h1) + (w2*h2) - ainter
    return ainter/aunion

print(recouvrement(10,10,30,30,25,25,30,30))

def exemplesNegatifs(n, data, pathTrain, newsize):
    images = np.zeros((len(data)*n,newsize,newsize))
    # Pour chaque image
    for i in range(len(data)):
        # On cherche n exemples négatifs
        for j in range(n):
            print("image",i, " (i:",i,", j:",j,", ite",(i*n) +j,")")
    return images

exNeg = exemplesNegatifs(10, data, pathTrain, newSize)

      
      