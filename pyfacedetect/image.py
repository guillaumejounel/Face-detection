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
def minFace(data) :
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
    img = np.array(Image.open(pathTrain + "%04d" %(data[n, 0]) + ".jpg"),
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
    minwh = np.minimum(data[:, 3], data[:, 4])
    # Modification de la largeur
    newData[:, 1] += (data[:, 3]-minwh)//2
    # Modification de la hauteur
    newData[:, 2] += (data[:, 4]-minwh)//2
    # Transformation des rectangles en carrés
    newData[:, 3:] = np.transpose(np.array([minwh, minwh]))
    return newData

# filtre gradient
def filtreLineaire(image):
    grad = np.gradient(image)
    return np.sqrt(grad[0]*grad[0]+grad[1]*grad[1])

# retourne un array avec les images croppées
def donneesImages(data, pathTrain, newsize):
    images = np.zeros((2*len(data),newsize,newsize))
    for i in range(len(data)):
        img = filtreLineaire(color.rgb2gray(cropImage(i, data, pathTrain, newsize)))
        images[2*i] = img
        images[2*i+1] = np.fliplr(img)
    return images

# gives an negative example from the n^th image
def negatifRandom(data,pathTrain,newsize,n):
    while True:
        # Récupération de l'image
        img = np.array(Image.open(pathTrain + "%04d" % (n) + ".jpg"),
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
    xinter = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
    yinter = max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
    ainter = xinter * yinter
    aunion = (w1*h1) + (w2*h2) - ainter
    return ainter/aunion

##############################################################################

# renvoie les coordonnées de la fenetre glissante suivante
# en fonction de n l'indice de l'image, les coordonnees de la fenetre actuelle
# le pas horizontal et le pas vertical
def fenetre_gliss_suiv(img, x, y, w, h, pas_hor, pas_vert, limite_x, limite_y):

    # on commence par le cas "trivial" : on décale la fenetre selon x
    x_next = x + pas_hor
    if (x_next + w) <= limite_x:
        return x_next, y

    # On doit revenir à la ligne
    x_next = 0
    y_next = y + pas_vert
    if y_next + h <= limite_y:
        return x_next, y_next

    # sinon, on a fini de parcourir l'image, on renvoie une erreur
    return -1, -1


# renvoie l'ensemble des fenêtres glissantes, triées par score
# renvoie un array de "fenetres" sous la forme score, x, y, w, h
def fenetre_glissante(clf, img, w, h, pas_hor, pas_vert, return_pos=1):
    img = color.rgb2gray(img)

    # on détermine les bordures de l'image
    limite_x = np.size(img, 1)
    limite_y = np.size(img, 0)

    # calcul du nombre de fenetres glissantes
    dim_x = int((limite_x - w) / pas_hor) + 1
    dim_y = int((limite_y - h) / pas_vert) + 1
    data = np.zeros((dim_x + 1) * (dim_y + 1), 5)

    # print("nb pas_x", dim_x, "nb pas_y", dim_y)

    indice = 0
    # on parcourt l'ensemble de l'image selon x, y
    for i in range(0, dim_x):
        for j in range(0, dim_y):
            x_tmp = i*pas_hor
            y_tmp = j*pas_vert
            img_tmp = img[y_tmp:y_tmp + h, x_tmp:x_tmp + w]
            img_tmp = np.reshape(img_tmp, (1, h*w))

            data[indice] = [clf.decision_function(img_tmp), x_tmp, y_tmp, w, h]
            indice += 1

    # derniere ligne selon y (x = cst)
    for i in range(0, dim_y):
        x_tmp = limite_x - w
        y_tmp = i * pas_vert
        img_tmp = img[y_tmp:y_tmp + h, x_tmp:x_tmp + w]
        img_tmp = np.reshape(img_tmp, (1, h*w))

        data[indice] = [clf.decision_function(img_tmp), x_tmp, y_tmp, w, h]

        indice += 1

    # derniere ligne selon x (y = cst
    for i in range(0, dim_x):
        x_tmp = i * pas_hor
        y_tmp = limite_y - h
        img_tmp = img[y_tmp:y_tmp + h, x_tmp:x_tmp + w]
        img_tmp = np.reshape(img_tmp, (1, h*w))

        data[indice] = [clf.decision_function(img_tmp), x_tmp, y_tmp, w, h]

        indice += 1

    if return_pos == 1:
        return data[data[:, 0] >= 1]
    else:
        return data


# affiche une image avec le rectangle associé
def afficher_fenetre_gliss(img, data_fenetre, pathTrain, only_pos=0):
    # Créer la figure et les axes
    fig,ax = plt.subplots(1)
    # Afficher l'image
    ax.imshow(img)
    # Créer le rectangle
    for i in range(1, np.size(data_fenetre, 0) + 1):
        score, xcorner, ycorner, width, height = data_fenetre[i-1]
        if (score >= 1) or (only_pos == 0):
            if score >= 1:
                color = 'g'
            else:
                color = 'r'

            rect = patches.Rectangle((xcorner, ycorner), width,
                                     height, linewidth=2, edgecolor=color,
                                     facecolor='none')
            # Ajouter le rectangle sur l'image
            ax.add_patch(rect)

    plt.show()


# supprime les boites "non maximales" (facteur de recouvrement)
def suppressionNonMaximas(data, facteur=0.3):
    # Tri des boîtes par ordre décroissant de score
    data.view('i8,i8,i8,i8,i8')[::-1].sort(order=['f0'], axis=0)
    i = 0
    # Pour chaque boite dans la liste
    while (i < len(data)):
        scorei, xi, yi, wi, hi = data[i]
        # On compare aux boites suivantes
        for j in range(i+1, len(data)):
            xj, yj, wj, hj = data[j][1:]
            print("Comparaison de",i,"et",j, "(recouvrement de ",recouvrement(xi,yi,wi,hi,xj,yj,wj,hj),")")
            # Si leur recouvrement est supérieur à 50% on ne les considère plus
            if recouvrement(xi,yi,wi,hi,xj,yj,wj,hj) > facteur:
                print("on ne garde pas",j)
                data[j][0] = 0
        # On passe à la boîte suivante
        i+=1
        # On saute les boîtes qui ont été éliminées
        while i < len(data) and data[i][0] == 0:
            i+=1
    # Retourne les boites qui n'ont pas été éliminées
    return data[data[:,0] != 0]