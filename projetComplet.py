# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
from skimage import io
from skimage import util
from skimage.transform import resize
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

def minFace(data):
    return int(np.min(data[:,3:]))

newSize = minFace(data)
print("minFace =", newSize)

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

def cropImage(n, data, pathTrain, newsize=0):
    img = np.array(Image.open(pathTrain +"%04d"%(data[n,0])+".jpg"), dtype=np.uint8)
    x, y, w, h = map(int, data[n][1:])
    img = img[y:y+h, x:x+w]
    if newsize:
        img = resize(img, (newsize, newsize), mode='reflect')
    return img

image = cropImage(14,data,pathTrain)
fig,ax = plt.subplots(1)
ax.imshow(image)
plt.show()


def dataSquare(data):
    # Récupération des "rectangles"
    newData = np.array(data)
    # Récupération du minimum entre la largeur et la hauteur
    minwh = np.minimum(data[:,3], data[:,4])
    # Modification de la largeur
    newData[:,1] += (data[:,3]-minwh)//2
    # Modification de la hauteur
    newData[:,2] += (data[:,4]-minwh)//2
    # Transformation des rectangles en carrés
    newData[:,3:] = np.transpose(np.array([minwh, minwh]))
    return newData

# Calcul des nouvelles coordonnées
dataPositif = dataSquare(data)

# Affichage des coordonnées sur une images
afficherImgRect(14, dataPositif, pathTrain)

#Recadrement et redimensionnement de l'image
image = cropImage(14,dataPositif,pathTrain, newSize)
fig,ax = plt.subplots(1)
ax.imshow(image)
plt.show()

def donneesImages(data, pathTrain, newsize):
    images = np.zeros((len(data),newsize,newsize))
    for i in range(len(data)):
        images[i] = color.rgb2gray(cropImage(i, data, pathTrain, newsize))
    return images

exemplesPositifs = donneesImages(dataPositif, pathTrain, newSize)
fig,ax = plt.subplots(1)
ax.imshow(exemplesPositifs[13])
plt.show()


img = np.array(Image.open(pathTrain +"%04d"%(14)+".jpg"), dtype=np.uint8)
def negatifRandom(data,pathTrain,newsize,n):
    while True:
        # Récupération de l'image
        img = np.array(Image.open(pathTrain +"%04d"%(n)+".jpg"), dtype=np.uint8)
        # Récupération de ses caractéristiques
        x1, y1, w1, h1 = map(int, data[n-1][1:])
        w, h = [len(img[0]), len(img)]
        # Choix aléatoire d'une taille de fenêtre
        taille = int(np.random.uniform(low=newsize, high=min(w,h)))
        # Choix aléatoire de la position de la fenêtre
        x = int(np.random.uniform(low=0, high=w-taille))
        y = int(np.random.uniform(low=0, high=h-taille))
        # Test du score de recouvrement de la fenêtre
        if recouvrement(x1,y1,w1,h1,x,y,taille,taille) < 0.5:
            return n, x, y, taille, taille

print(negatifRandom(data,pathTrain,newSize,14))

def exemplesNegatifs(n, data, pathTrain, newsize):
    newData = np.zeros((len(data)*n, 5))
    # Pour chaque image
    for i in range(len(data)):
        # On cherche n exemples négatifs
        for j in range(n):
            newData[i*n+j,:] = negatifRandom(data,pathTrain,newsize,i+1)
    return newData

dataNegatif = exemplesNegatifs(10, data, pathTrain, newSize)

exemplesNegatifs = donneesImages(dataNegatif, pathTrain, newSize)

fig,ax = plt.subplots(1)

ax.imshow(exemplesNegatifs[13])
plt.show()
