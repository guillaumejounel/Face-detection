# -*- coding: utf-8 -*-

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

import time
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
print("Taille des carrés :", newSize)

# Calcul des nouvelles datas et de coordonnées
dataPositif = libimg.dataSquare(data, pathTrain)

#Visualisation d'une image et de son filtre linéaire
#io.imshow(libimg.cropImage(7,dataPositif,pathTrain,newSize))
#libimg.filtreLineaire(color.rgb2gray(libimg.cropImage(7,dataPositif,pathTrain,newSize)), s=9, visualisation=1)

# Calcul de la taille du descripteur
tailleDescripteur = len(libimg.filtreLineaire(color.rgb2gray(libimg.cropImage(7,dataPositif,pathTrain,newSize)), s=9))
print("Taille du descripteur :", tailleDescripteur)


# on calcul le nouveau set d'image (avec application de filtre)
print("\n-- Calcul des vecteurs des images positives --")
exemplesPositifs = libimg.donneesImages(dataPositif, pathTrain, newSize,
                                        tailleDescripteur, etat=1)

# Génération des exemples négatifs (nb_neg par images)
print("\n-- Calcul du set d'images negatives --")
factor_neg = 10
print(" -> Calcul des coordonnées de",factor_neg,"négatifs par image...")
dataNegatif = libimg.exemplesNegatifs(factor_neg, data, pathTrain,
                                      newSize, maxrecouvrement=0.2, etat=1)
print(" -> Calcul des",len(dataNegatif),"vecteurs descripteurs négatifs...")
exemplesNegatifs = libimg.donneesImages(dataNegatif, pathTrain, newSize,
                                        tailleDescripteur, etat=1)
print("-- Génération d'exemple terminée ! --")

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Création des données du classifieur

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

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Création des données du classifieur
result_c = np.zeros((5,3))

N = 1

for i in range(0,5):
    result_c[i][0] = i + N
    
    print("calcul pour c = ", i + N)
    
    clf = svm.SVC(kernel='linear', C=i + N)
    clf.fit(exemples,y)
    
    dataCalc = libimg.calculResultats(clf, 1, pathTest, newSize,
                                      tailleDescripteur, fin = 100, etat=1)
    
    result_train =  libimg.calculResultatsTrain(clf, data, dataCalc)
    result_c[i][1], result_c[i][2] = libimg.affichAnalyseResultat(result_train,
                                                                 data)


plt.plot(result_c[:,1]*100)
plt.ylabel('Rappel')
plt.xlabel('C')
plt.show()

plt.plot(result_c[:,2]*100)
plt.ylabel('Précision')
plt.xlabel('C')
plt.show()
