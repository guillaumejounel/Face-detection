import numpy as np
from skimage import io, util, exposure
from skimage.transform import resize
from skimage import color
#from skimage import feature
from sklearn import svm, preprocessing
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from skimage.feature import hog
from skimage import feature

taille_descripteur = 72
seuil_validation = 0

# retourne la taille minimale d'un visage
def minFace(data) :
    return int(np.min(data[:,3:]))


# affiche une image avec le rectangle associé
def afficherImgRect(n, data, pathTrain):
    # Charger l'image
    img = np.array(io.imread(pathTrain +"%04d"%(n)+".jpg"), dtype=np.uint8)
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
    img = np.array(io.imread(pathTrain + "%04d" %(data[n, 0]) + ".jpg"),
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
    #grad = np.gradient(image)
    #return np.sqrt(grad[0]*grad[0]+grad[1]*grad[1])
    return preprocessing.minmax_scale(hog(image, 8, [10,10], [1,1], False, True))

# Descripteur Multi-block Local Binary Patterns
# http://www.cbsr.ia.ac.cn/users/scliao/papers/Zhang-ICB07-MBLBP.pdf
def MBLBPdescriptor(img, pas=0.05, animated=0):
    desc = np.zeros((int(np.square((1-0.1)/pas)), 1))
    i = 0
    for rx in np.arange(0.1,1,0.05):
        for ry in np.arange(0.1,1,0.05):
            h, w = img.shape
            hr, wr = int(h*ry), int(w*rx)
            if animated:
                fig, ax = plt.subplots(1)
                plt.imshow(feature.draw_multiblock_lbp(img, (h-hr)//2, (w-wr)//2, wr//3, hr//3))
                plt.show()
                print(i)
            desc[i] = feature.multiblock_lbp(img, (h-hr)//2, (w-wr)//2, wr//3, hr//3)
            i+=1
    return desc

# retourne un array avec les images croppées
def donneesImages(data, pathTrain, newsize):
    taille = newsize*newsize
    images = np.zeros((2*len(data),taille_descripteur))
    for i in range(len(data)):
        img = color.rgb2gray(cropImage(i, data, pathTrain, newsize))
        images[2*i] = filtreLineaire(img).reshape(taille_descripteur,)
        images[2*i+1] = filtreLineaire(np.fliplr(img)).reshape(taille_descripteur,)
    return images

# gives an negative example from the n^th image
def negatifRandom(data,pathTrain,newsize,n):
    while True:
        # Récupération de l'image
        img = np.array(io.imread(pathTrain + "%04d" % (n) + ".jpg"),
                       dtype=np.uint8)
        # Récupération de ses caractéristiques
        x1, y1, w1, h1 = map(int, data[n-1][1:])
        h, w = img.shape[:2]
        # Choix aléatoire d'une taille de fenêtre
        taille = int(np.random.uniform(low=newsize, high=min(w,h)))
        # Choix aléatoire de la position de la fenêtre
        x = int(np.random.uniform(low=0, high=w-taille))
        y = int(np.random.uniform(low=0, high=h-taille))
        # Test du score de recouvrement de la fenêtre
        if recouvrement(x1,y1,w1,h1,x,y,taille,taille) < 0.3:
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


# renvoie l'ensemble des fenêtres glissantes, triées par score
# renvoie un array de "fenetres" sous la forme score, x, y, w, h
def fenetre_glissante(clf, img, ratio, w, h, pas_hor, pas_vert, return_pos=1):
    img = color.rgb2gray(img)

    # on détermine les bordures de l'image
    limite_y, limite_x = img.shape

    # calcul du nombre de fenetres glissantes
    dim_x = int((limite_x - w) / pas_hor) + 1
    dim_y = int((limite_y - h) / pas_vert) + 1
    data = np.zeros(((dim_x + 1) * (dim_y + 1) -1, 5))

    # print("nb pas_x", dim_x, "nb pas_y", dim_y)

    indice = 0
    # on parcourt l'ensemble de l'image selon x, y
    for i in range(0, dim_x):
        for j in range(0, dim_y):
            x_tmp = i*pas_hor
            y_tmp = j*pas_vert
            img_tmp = img[y_tmp:y_tmp + h, x_tmp:x_tmp + w]
            img_tmp = filtreLineaire(img_tmp).reshape(taille_descripteur,)
            data[indice] = [clf.decision_function(img_tmp), x_tmp, y_tmp, w, h]
            indice += 1

    # derniere ligne selon y (x = cst)
    for i in range(0, dim_y):
        x_tmp = limite_x - w
        y_tmp = i * pas_vert
        img_tmp = img[y_tmp:y_tmp + h, x_tmp:x_tmp + w]
        #img_tmp = np.reshape(img_tmp, (1, h*w))
        img_tmp = filtreLineaire(img_tmp).reshape(taille_descripteur,)
        data[indice] = [clf.decision_function(img_tmp), x_tmp, y_tmp, w, h]

        indice += 1

    # derniere ligne selon x (y = cst
    for i in range(0, dim_x):
        x_tmp = i * pas_hor
        y_tmp = limite_y - h
        img_tmp = img[y_tmp:y_tmp + h, x_tmp:x_tmp + w]
        img_tmp = filtreLineaire(img_tmp).reshape(taille_descripteur,)
        #img_tmp = np.reshape(img_tmp, (1, h*w))

        data[indice] = [clf.decision_function(img_tmp), x_tmp, y_tmp, w, h]

        indice += 1
    
    #On remetà la taille de l'image d'origine
    data[:,1:] /= ratio
    if return_pos == 1:
        return data[data[:, 0] >= seuil_validation]
    else:
        return data


# affiche une image avec le rectangle associé
def afficher_fenetre_gliss(img, data_fenetre, pathTrain, scoremin, only_pos=0,animated=0):
    detections = 0
    if animated==0:
        # Créer la figure et les axes
        fig,ax = plt.subplots(1)
        # Afficher l'image
        ax.imshow(img)
        # Créer le rectangle
    for i in range(1, np.size(data_fenetre, 0) + 1):
        if animated==1:
            # Créer la figure et les axes
            fig,ax = plt.subplots(1)
            # Afficher l'image
            ax.imshow(img)
            # Créer le rectangle
        score, xcorner, ycorner, width, height = data_fenetre[i-1]
        if (score >= scoremin) or (only_pos == 0):
            if score >= 0:
                color = 'g'
                detections+=1
            else:
                color = 'r'

            rect = patches.Rectangle((xcorner, ycorner), width,
                                     height, linewidth=2, edgecolor=color,
                                     facecolor='none')
            # Ajouter le rectangle sur l'image
            ax.add_patch(rect)
            if animated==1:
                plt.show()
        if animated==1:
            print("Animation en cours... ",detections,"boîtes détectées !")
    if animated==0:
        plt.show()


# supprime les boites "non maximales" (facteur de recouvrement)
def suppressionNonMaximas(data, facteur=0.2):
    # Tri des boîtes par ordre décroissant de score
    data.view('i8,i8,i8,i8,i8')[::-1].sort(order=['f0'], axis=0)
    i = 0
    # Pour chaque boite dans la liste
    while (i < len(data)):
        scorei, xi, yi, wi, hi = data[i]
        # On compare aux boites suivantes
        for j in range(i+1, len(data)):
            if data[j][0] != 0:
                xj, yj, wj, hj = data[j][1:]
                #print("Comparaison de",i,"et",j, "(recouvrement de ",recouvrement(xi,yi,wi,hi,xj,yj,wj,hj),")")
                # Si leur recouvrement est supérieur à 50% on ne les considère plus
                if recouvrement(xi,yi,wi,hi,xj,yj,wj,hj) > facteur:
                    #print("on ne garde pas",j)
                    data[j][0] = 0
                    # Test : augmentation du score si recouvrement
                    #data[i][0] += 0.1
        # On passe à la boîte suivante
        i+=1
        # On saute les boîtes qui ont été éliminées
        #while i < len(data) and data[i][0] == 0:
        #    i+=1
    # Retourne les boites qui n'ont pas été éliminées
    return data[data[:,0] != 0]


def fenetre_glissante_multiechelle(clf, img):
    data = np.zeros((1000, 5)) # Max 100 fenêtres
    cursor = 0
    for ratio in np.arange(30/min(img.shape), 0.6, 0.1):
        #print("Fenêtre glissante",round(ratio*100),"%")
        data_f = fenetre_glissante(clf, rescale(img, ratio, mode='reflect'), ratio, 30, 30, 10,10, return_pos=1)
        for i in range(len(data_f)):
            data[i+cursor] = data_f[i]
        cursor += len(data_f)
    return suppressionNonMaximas(data)


def fauxPositifs(clf, pathTrain, data):
    fpos = np.zeros((10000, 5))
    cursor = 0
    for i in range(len(data)):
        xi, yi, wi, hi = map(int, data[i][1:])
        print("Calcul faux positifs :",i//10,"% (", len(fpos[fpos[:,0]!=0]),")")
        img = np.array(io.imread(pathTrain +"%04d"%(i+1)+".jpg", as_grey=True))
        data_f = fenetre_glissante_multiechelle(clf, img)
        for j in range(len(data_f)):
            xj, yj, wj, hj = map(int, data_f[j][1:])
            if recouvrement(xj, yj, wj, hj, xi, yi, wi, hi) < 0.3:
                fpos[cursor] = data_f[j]
                fpos[cursor, 0] = i+1
                cursor += 1
    return fpos