 #-*- coding: utf-8 -*-

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# imports et définition des paths
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
import pyfacedetect.image as libimg

# changer le chemin de IPython
# avec la commande "%bookmark  NOM_DU_MARQUE_PAGE /path/to/dir"
# %bookmark PROJET /Users/guillaume/Cloud/WORK/UTC/GI02/SY32/TDXu/Projet/
# (à ne faire qu'une fois, normalement c'est persistant)
# puis lancer la commande "%cd -b PROJET" en début de session.

pathTrain = "projetface/train/"
pathTest =  "projetface/test/"
pathFile = "projetface/label.txt"
data = np.loadtxt(pathFile)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# génération des exemples

# détermination de la taille des carrés
newSize = libimg.minFace(data)
print("taille des carrés : ", newSize)

# Calcul des nouvelles datas et de coordonnées
dataPositif = libimg.dataSquare(data)

# on calcul le nouvel set d'image (en noir & blanc)
print("Calcul du set d'image positif")
exemplesPositifs = libimg.donneesImages(dataPositif, pathTrain, newSize)

# Génération des exemples négatifs (nb_neg par images)
print("Calcul du set d'image negatif")
factor_neg = 1
dataNegatif = libimg.exemplesNegatifs(factor_neg, data, pathTrain, newSize)
exemplesNegatifs = libimg.donneesImages(dataNegatif, pathTrain, newSize)

print("Génération d'exemple terminée !")

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Création du classifieur

# TODO nb dynamiques
nb_pos = 1000
nb_neg = factor_neg * nb_pos

# concaténation des exemples
exemples = np.concatenate((exemplesPositifs, exemplesNegatifs), axis=1)
exemples = np.reshape(exemples,(nb_pos + nb_neg, 900))

# vérification
# exemples = np.reshape(exemples,(nb_pos + nb_neg, 30,30))
# plt.figure(1)
# plt.imshow(exemples[14])
# plt.show()

y = np.concatenate((np.ones(nb_pos), np.zeros(nb_neg)))

clf = svm.SVC(kernel='linear')
clf.fit(exemples,y)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# test du classifieur

print(np.mean(clf.predict(exemples) != y)*100)
