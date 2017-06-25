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
# %load_ext autoreload
# %autoreload 2
# %aimport pyfacedetect.image
# %aimport pyfacedetect.learn

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
print("\n-- Calcul du set d'images positives --")
exemplesPositifs = libimg.donneesImages(dataPositif, pathTrain, newSize, tailleDescripteur, etat=1)

# Génération des exemples négatifs (nb_neg par images)
print("\n-- Calcul du set d'images negatives --")
factor_neg = 10
print(" -> Calcul des coordonnées de",factor_neg,"négatifs par image...")
dataNegatif = libimg.exemplesNegatifs(factor_neg, data, pathTrain, newSize, maxrecouvrement=0.2, etat=1)
print(" -> Calcul des",len(dataNegatif),"vecteurs descripteurs négatifs...")
exemplesNegatifs = libimg.donneesImages(dataNegatif, pathTrain, newSize, tailleDescripteur, etat=1)
print("-- Génération d'exemple terminée ! --")

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

print("\n-- Création du classifieur SVM --")
#clf = AdaBoostClassifier()
clf = svm.SVC(kernel='linear', C=7.1)
print(" -> Apprentissage initial...")
clf.fit(exemples,y)
# Optimisation du classifieur
print(" -> Validation croisée...")
print('   - Résultat :', liblearn.validationCroisee(clf, exemples, y, 5))
print("-- Fin de la création du classifieur --")

# AdaBoostClassifier() -> 4.74
# AdaBoostClassifier(n_estimators=100) -> 4.40
# AdaBoostClassifier(n_estimators=100, learning_rate=1.5) -> 5.11
# AdaBoostClassifier(n_estimators=100, learning_rate=0.5) -> 4.17
# AdaBoostClassifier(n_estimators=100, learning_rate=0.3) -> 4.74

# svm.SVC(kernel='linear') -> 4.34
# svm.SVC(kernel='linear', C=5) -> 4.32
# svm.SVC(kernel='linear', C=7.1) -> 3.30


#liblearn.graphValidationCroisee(clf, exemples, 6.5, 7.5, 0.1)
# C=7.1 pas mal !


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


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# entrainement sur les faux positifs (très très long)

print("\n-- Création de faux positifs --")

print(" -> Calcul des coordonnées de faux positifs...")
# mon ordinateur ne supporte pas le calcul des 1000 d'un coup (env 7 heures): à faire de 100 en 100...
#dataFp = libimg.fauxPositifs(clf, pathTrain, data, 0, 1000, 0.5, newSize, tailleDescripteur, etat=1)
dataFp = libimg.fauxPositifs(clf, pathTrain, data, 0, 100, 0.5, newSize, tailleDescripteur, etat=1)
dataFp = np.concatenate((dataFp, libimg.fauxPositifs(clf, pathTrain, data, 100, 200, 0.5, newSize, tailleDescripteur, etat=1)), axis=0)
dataFp = np.concatenate((dataFp, libimg.fauxPositifs(clf, pathTrain, data, 200, 300, 0.5, newSize, tailleDescripteur, etat=1)), axis=0)
dataFp = np.concatenate((dataFp, libimg.fauxPositifs(clf, pathTrain, data, 300, 400, 0.5, newSize, tailleDescripteur, etat=1)), axis=0)
dataFp = np.concatenate((dataFp, libimg.fauxPositifs(clf, pathTrain, data, 400, 500, 0.5, newSize, tailleDescripteur, etat=1)), axis=0)
dataFp = np.concatenate((dataFp, libimg.fauxPositifs(clf, pathTrain, data, 500, 600, 0.5, newSize, tailleDescripteur, etat=1)), axis=0)
dataFp = np.concatenate((dataFp, libimg.fauxPositifs(clf, pathTrain, data, 600, 700, 0.5, newSize, tailleDescripteur, etat=1)), axis=0)
dataFp = np.concatenate((dataFp, libimg.fauxPositifs(clf, pathTrain, data, 700, 800, 0.5, newSize, tailleDescripteur, etat=1)), axis=0)
dataFp = np.concatenate((dataFp, libimg.fauxPositifs(clf, pathTrain, data, 800, 900, 0.5, newSize, tailleDescripteur, etat=1)), axis=0)
dataFp = np.concatenate((dataFp, libimg.fauxPositifs(clf, pathTrain, data, 900, 1000, 0.5, newSize, tailleDescripteur, etat=1)), axis=0)
# SAUVEGARDER la variable après chaque 100 !

print(" -> Calcul des",len(dataFp),"vecteurs descripteurs...")
exFp = libimg.donneesImages(dataFp, pathTrain, newSize, tailleDescripteur)

exemplesNegatifs = np.concatenate((exemplesNegatifs, exFp), axis=0)
nb_pos = exemplesPositifs.shape[0]
nb_neg = exemplesNegatifs.shape[0]
y = np.concatenate((np.ones(nb_pos), np.zeros(nb_neg)))
# concaténation des exemples
exemples = np.concatenate((exemplesPositifs, exemplesNegatifs), axis=0)

print("-- Fin de la création des faux positifs --")

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

print("\n-- Création du nouveau classifieur --")
#clf = AdaBoostClassifier()
#clf = RandomForestClassifier()
clf = svm.SVC(kernel='linear', C=7.1)
print(" -> Apprentissage...")
clf.fit(exemples,y)

#Recherche du meilleur C : graphe
#liblearn.graphValidationCroisee(clf, exemples, 8.5, 9.5, 0.1)

print(" -> Validation croisée...")
print('validation croisée :', liblearn.validationCroisee(clf, exemples, y, 5))

# svm.SVC(kernel='linear', C=7.1) -> 5.79

# TODO? Ré-optimisation de C ? 
print("-- Fin de la création du classifieur --")

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

img = np.array(io.imread(pathTest +"%04d"%(146)+".jpg", as_grey=True))
data_f = libimg.fenetre_glissante_multiechelle(clf, 1, img, newSize, tailleDescripteur, animated=0, return_pos=1)
libimg.afficher_fenetre_gliss(img, data_f, 1, only_pos=0, animated=0)

   
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Calcul des résultats pour les images de test

dataCalc = libimg.calculResultats(clf, 0, pathTest, 447, newSize, tailleDescripteur, etat=1)


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Voir quelques résultats

nbim = 30
img = np.array(io.imread(pathTest +"%04d"%(nbim)+".jpg"), dtype=np.uint8)
fig,ax = plt.subplots(1)
ax.imshow(img)
for rectangle in dataCalc[dataCalc[:,0]==nbim]:
    xcorner, ycorner, width, height = rectangle[1:5]
    rect = patches.Rectangle((xcorner, ycorner), width, height,
                             linewidth=2,edgecolor='r', facecolor='none')
    ax.add_patch(rect)
plt.show()

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Sauvegarde des résultats
np.savetxt("result.txt", dataCalc, fmt="%03i %i %i %i %i %0.2f")
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

