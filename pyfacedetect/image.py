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

# retourne la taille minimale d'un visage
def minFace(data):
    return int(np.min(data[:,3:]))

# affiche une image avec le rectangle associé
def afficherImgRect(n, data, pathTrain):
    # Charger l'image
    img = np.array(Image.open(pathTrain +"%04d"%(n)+".jpg"), dtype=np.uint8)
    # Créer la figure et les axes
    fig,ax = plt.subplots(1)
    # Afficher l'image
    ax.imshow(img)
    # Créer le rectangle
    xcorner, ycorner, width, height = data[n-1][1:]
    rect = patches.Rectangle((xcorner, ycorner), width, height,
                             linewidth=2,edgecolor='r', facecolor='none')
    # Ajouter le rectangle sur l'image
    ax.add_patch(rect)
    plt.show()

# retourne l'image croppé selon le rectancle
# si l'argument newsize vaut 1 / True croppe selon un carré
def cropImage(n, data, pathTrain, newsize=0):
    img = np.array(Image.open(pathTrain +"%04d"%(data[n,0])+".jpg"),
                   dtype=np.uint8)
    x, y, w, h = map(int, data[n][1:])
    img = img[y:y+h, x:x+w]
    if newsize:
        img = resize(img, (newsize, newsize), mode='reflect')
    return img

# Transforme les rectangles des datas en carrés
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

# retourne un array avec les images croppées
def donneesImages(data, pathTrain, newsize):
    images = np.zeros((len(data),newsize,newsize))
    for i in range(len(data)):
        images[i] = color.rgb2gray(cropImage(i, data, pathTrain, newsize))
    return images

# gives an negative example from the n^th image
def negatifRandom(data,pathTrain,newsize,n):
    while True:
        # Récupération de l'image
        img = np.array(Image.open(pathTrain +"%04d"%(n)+".jpg"),
                       dtype=np.uint8)
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

# renvoie pour chaque image n exemples négatifs
def exemplesNegatifs(n, data, pathTrain, newsize):
    newData = np.zeros((len(data)*n, 5))
    # Pour chaque image
    for i in range(len(data)):
        # On cherche n exemples négatifs
        for j in range(n):
            newData[i*n+j,:] = negatifRandom(data,pathTrain,newsize,i+1)
    return newData

# revoie le recouvrement de deux carrés
def recouvrement(x1, y1, w1, h1, x2, y2, w2, h2):
    xinter = max(0, min(x1+w1,x2+w2) - max(x1,x2))
    yinter = max(0, min(y1+h1,y2+h2) - max(y1,y2));
    ainter = xinter * yinter;
    aunion = (w1*h1) + (w2*h2) - ainter
    return ainter/aunion
