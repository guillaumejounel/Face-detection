 #-*- coding: utf-8 -*-

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# imports et définition des paths
import numpy as np
from skimage import io
from skimage import util
from skimage.transform import resize
from skimage import color
from skimage import feature
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from PIL import Image
import pyfacedetect.image as libimg
import pyfacedetect.learn as liblearn
from skimage.transform import rescale

# changer le chemin de IPython
# avec la commande "%bookmark  NOM_DU_MARQUE_PAGE /path/to/dir"
# %bookmark PROJET /Users/guillaume/Cloud/WORK/UTC/GI02/SY32/TDXu/Projet/
# (à ne faire qu'une fois, normalement c'est persistant)
# puis lancer la commande "%cd -b PROJET" en début de session.

import warnings

warnings.filterwarnings('ignore')

pathTrain = "projetface/train/"
pathTest = "projetface/test/"
pathFile = "projetface/label.txt"
data = np.loadtxt(pathFile)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
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
factor_neg = 10
dataNegatif = libimg.exemplesNegatifs(factor_neg, data, pathTrain, newSize)
exemplesNegatifs = libimg.donneesImages(dataNegatif, pathTrain, newSize)

print("Génération d'exemple terminée !")

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Création du classifieur

# nb dynamiques
nb_pos = exemplesPositifs.shape[0]
nb_neg = exemplesNegatifs.shape[0]

# concaténation des exemples
exemples = np.concatenate((exemplesPositifs, exemplesNegatifs), axis=0)
#exemples = np.reshape(exemples,(nb_pos + nb_neg, 900))

# vérification si l'on veut afficher les images de l'array exemples
# exemples = np.reshape(exemples,(nb_pos + nb_neg, 30,30))
# plt.figure(1)
# plt.imshow(exemples[14])
# plt.show()

y = np.concatenate((np.ones(nb_pos), -np.ones(nb_neg)))

print("Création du classifieur et entrainement initial")
clf = AdaBoostClassifier()
clf.fit(exemples,y)

# test du classifieur

print(np.mean(clf.predict(exemples) != y)*100)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# entrainement sur les faux positifs

dataFp = libimg.fauxPositifs(clf, pathTrain, data)
exFp = libimg.donneesImages(dataFp, pathTrain, newSize)
exemplesNegatifs = np.concatenate((exemplesNegatifs, exFp), axis=0)

nb_pos = exemplesPositifs.shape[0]
nb_neg = exemplesNegatifs.shape[0]

y = np.concatenate((np.ones(nb_pos), np.zeros(nb_neg)))
# concaténation des exemples
exemples = np.concatenate((exemplesPositifs, exemplesNegatifs), axis=0)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

print("Création du nouveau classifieur")
clf = AdaBoostClassifier()
clf.fit(exemples,y)

#Problème : le meilleur score c'est quand on ne détecte rien...
# (il faut utiliser les "courbes" rappel/précision vues en cours)

#print('validation croisée :', validationCroisee(clf, exemples, y, 5))
# AdaBoostClassifier() -> 4.74
# AdaBoostClassifier(n_estimators=100) -> 4.40
# AdaBoostClassifier(n_estimators=100, learning_rate=1.5) -> 5.11
# AdaBoostClassifier(n_estimators=100, learning_rate=0.5) -> 4.17
# AdaBoostClassifier(n_estimators=100, learning_rate=0.3) -> 4.74

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

dataCalc = calculResultats(clf, pathTrain, data)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
