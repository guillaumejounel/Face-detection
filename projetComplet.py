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

# vérification
# exemples = np.reshape(exemples,(nb_pos + nb_neg, 30,30))
# plt.figure(1)
# plt.imshow(exemples[14])
# plt.show()

# TODO 1 ou -1 !
y = np.concatenate((np.ones(nb_pos), np.zeros(nb_neg)))

print("Création du classifieur")
clf = svm.SVC()
clf.fit(exemples,y)

# test du classifieur

print(np.mean(clf.predict(exemples) != y)*100)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%




dataFp = fauxPositifs(clf, pathTrain, data)
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
#clf = svm.SVC(kernel='linear', C=1000)
clf.fit(exemples,y)

#Problème : le meilleur score c'est quand on ne détecte rien...
# (il faut utiliser les "courbes" rappel/précision vues en cours)

#print('validation croisée :', validationCroisee(clf, exemples, y, 5))

# AdaBoostClassifier() -> 4.74
# AdaBoostClassifier(n_estimators=100) -> 4.40
# AdaBoostClassifier(n_estimators=100, learning_rate=1.5) -> 5.11
# AdaBoostClassifier(n_estimators=100, learning_rate=0.5) -> 4.17
# AdaBoostClassifier(n_estimators=100, learning_rate=0.3) -> 4.74

# svm.SVC(kernel='linear') -> 4.34
# svm.SVC(kernel='linear', C=5) -> 4.32
# svm.SVC(kernel='linear', C=1000) -> 

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

#The usual way to adjust the C parameter is by a grid search. Set a range of
# feasible values for C, for instance C in [0,15]. Then make a coarse search
# in laps of 1: 1,2,3,4,...,15. Look for the average error using a 5 or 10
# fold cross validation using the training set and keep the best value.
# Then perform the same procedure but on a finer search. For instance, say
# the best value (less error) was for C=5. Now look on laps of 0.1 in the
# range [4.1,5.9].

#C is a trade-off between training error and the flatness of the solution.
# The larger C is the less the final training error will be. But if you
# increase C too much you risk losing the generalization properties of the
# classifier, because it will try to fit as best as possible all the training
# points (including the possible errors of your dataset). In addition a large
# C, usually increases the time needed for training. 

np.array()
for i in range(16):
    clf = svm.SVC(kernel='linear', C=i)
    clf.fit(exemples,y)
    

img = np.array(io.imread(pathTest +"%04d"%(153)+".jpg", as_grey=True))
data_f = fenetre_glissante_multiechelle(clf, img)
afficher_fenetre_gliss(img, data_f, pathTest, 0, only_pos=0,animated=0)
    
dataCalc = calculResultats(clf, pathTest, 447)


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


#Voir les rectangles obtenus pour l'image n°nbim
nbim = 45
img = np.array(io.imread(pathTest +"%04d"%(nbim)+".jpg"), dtype=np.uint8)
fig,ax = plt.subplots(1)
ax.imshow(img)
for rectangle in dataCalc[dataCalc[:,0]==nbim]:
    xcorner, ycorner, width, height = rectangle[1:5]
    rect = patches.Rectangle((xcorner, ycorner), width, height,
                             linewidth=2,edgecolor='r', facecolor='none')
    ax.add_patch(rect)
plt.show()

#Sauvegarde des résultats
np.savetxt("ok.txt", dataCalc, fmt="%03i %i %i %i %i %0.2f")
#exemple : 001 20 32 64 128 0.23

#AdaBoostClassifier() : 2017-06-22 13:01:41	jounel	28.96	23.71	26.08	10.17


# TODO POUR LE RAPPORT :
    
#Choix des paramètres (à justifier pour le projet) :

#- Taille de la fenêtre glissante (plus petit visage, on préfère réduire la taille de l'image que zoomer)
#- Nombre d'exemples négatifs aléatoires (+ il y en a, plus c'est performant, mais coût ! ~10-20)
#- Taille du pas (spatial et échelle) - relatif à la taille de l'image
#- choix du vecteur descripteur
#- choix du classifieur (et de ses paramètres)

#Il faut aussi transformer les données de rectangles en carrés (en prenant le centre par exemple, attention il ne faut pas sortir de l'image)

# Score des boîtes entre 0 et +\infty (distance à la frontière).


        