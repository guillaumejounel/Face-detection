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
from skimage.transform import rescale
import os

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Récupération des images


# retourne la taille du plus petit visage des datas
def minFace(data, interieur=0) :
    if interieur:
        return int(np.min(data[:,3:]))
    return int(min(np.maximum(data[:, 3], data[:, 4])))


# affiche une image avec le rectangle associé
def afficherImgRect(n, data, pathTrain):
    # Charger l'image
    img = np.array(io.imread(pathTrain +"%04d"%(data[n][0])+".jpg"), dtype=np.uint8)
    # Créer la figure et les axes
    fig,ax = plt.subplots(1)
    # Afficher l'image
    ax.imshow(img)
    # Créer le rectangle
    xcorner, ycorner, width, height = data[n][1:5]
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
    x, y, w, h = map(int, data[n][1:5])
    img = img[y:y+h, x:x+w]
    if newsize:
        img = resize(img, (newsize, newsize), mode='reflect')
    return img

# Transforme les rectangles des datas en carrés
def dataSquare(data, path, interieur=0):
    # Récupération des "rectangles"
    newData = np.array(data)
    if interieur:
        # Récupération du minimum entre la largeur et la hauteur
        minwh = np.minimum(data[:, 3], data[:, 4])
        
        # Modification de la largeur
        newData[:, 1] += (data[:, 3]-minwh)//2
               
        # Modification de la hauteur
        newData[:, 2] += (data[:, 4]-minwh)//2
               
        # Transformation des rectangles en carrés
        newData[:, 3:] = np.transpose(np.array([minwh, minwh]))
    
    else:
        #Récupération de la taille des images
        imgsize = np.zeros((len(data), 2))
        for i in range(len(data)):
            imgsize[i] = io.imread(path + "%04d" %(data[1, 0]) + ".jpg").shape[:2]
        
        # Récupération du maximum entre la largeur et la hauteur
        maxwh = np.maximum(data[:, 3], data[:, 4])
        
        # Modification de la largeur
        newData[:, 1] -= (maxwh-data[:, 3])//2
        
        # Modification de la hauteur
        newData[:, 2] += (maxwh-data[:, 4])//2
               
        # Transformation des rectangles en carrés
        newData[:, 3:] = np.transpose(np.array([maxwh, maxwh]))
        
        return newData[(newData[:,1] >= 0) & (newData[:,1] < imgsize[:,1]) & (newData[:,2] >= 0) & (newData[:,2] < imgsize[:,0])]
    
    return newData

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Création du classifieur


# filtre gradient
def filtreLineaire(image, s=9, visualisation=0):
    #grad = np.gradient(image)
    #return np.sqrt(grad[0]*grad[0]+grad[1]*grad[1])
    if visualisation:
        io.imshow(preprocessing.minmax_scale(hog(image, 8, [s,s], [1,1], 
                                                 #block_norm='L1', 
                                                 visualise=True)[1]))
    return preprocessing.minmax_scale(hog(image, 8, [s,s], [1,1],
                                          #block_norm='L1'
                                          ))


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

# retourne un array avec les descripteurs correspondant à chaque image
# images = [ [image1            desc[0] desc[1] ... desc[taille_descripteur]
#            [symétrique_image1 desc[0] desc[1] ... desc[taille_descripteur]
#            [image 2 ....]]
def donneesImages(data, pathTrain, newsize, tailleDescripteur,etat=0):
    # taille = newsize*newsize
    images = np.zeros((2*len(data),tailleDescripteur))
    for i in range(len(data)):
        img = color.rgb2gray(cropImage(i, data, pathTrain, newsize))
        images[2*i] = filtreLineaire(img).reshape(tailleDescripteur,)
        images[2*i+1] = filtreLineaire(np.fliplr(img)).reshape(tailleDescripteur,)
        if etat:
            pct = round(100*(i/len(data)))
            print("\r"+str(pct//2*"-"+"{}%".format(pct)), end="\r")
    if etat:
        print()
    return images

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Génération des exemples négatifs

# gives an negative example from the n^th image
def negatifRandom(data, pathTrain, newsize, maxrecouvrement=0.2):
    # Récupération de l'image et de ses caractéristiques
    img = np.array(io.imread(pathTrain + "%04d" % (data[0]) + ".jpg"),
                   dtype=np.uint8)
    # Récupération de ses caractéristiques
    x1, y1, w1, h1 = map(int, data[1:])
    h, w = img.shape[:2]
    # on choisit des images aléatoires dans l'image jusqu'à ce que l'on en
    # trouve une qui ne soit pas le visage (recouvrement < 0.3)
    while True:
        # Choix aléatoire d'une taille de fenêtre
        taille = int(np.random.uniform(low=newsize, high=min(w,h)))
        # Choix aléatoire de la position de la fenêtre
        x = int(np.random.uniform(low=0, high=w-taille))
        y = int(np.random.uniform(low=0, high=h-taille))
        # Test du score de recouvrement de la fenêtre
        if recouvrement(x1,y1,w1,h1,x,y,taille,taille) < maxrecouvrement:
            return data[0], x, y, taille, taille


# renvoie pour chaque image n exemples négatifs
def exemplesNegatifs(n, data, pathTrain, newsize, maxrecouvrement=0.2, etat=0):
    newData = np.zeros((len(data)*n, 5))
    # Pour chaque image
    for i in range(len(data)):
        # On cherche n exemples négatifs
        for j in range(n):
            newData[i*n+j,:] = negatifRandom(data[i], pathTrain, newsize, maxrecouvrement)
            if etat:
                pct = round((100*(i*n+j))/(len(data)*n))
                print("\r"+str(pct//2*"-"+"{}%".format(pct)), end="\r")     
    if etat:
        print()
    return newData

# revoie le recouvrement de deux carrés
def recouvrement(x1, y1, w1, h1, x2, y2, w2, h2):
    xinter = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
    yinter = max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
    ainter = xinter * yinter
    aunion = (w1*h1) + (w2*h2) - ainter
    return ainter/aunion

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Fenêtre glissante


# renvoie l'ensemble des fenêtres glissantes, triées par score
# renvoie un array de "fenetres" sous la forme score, x, y, w, h
# si return_pos = 1 (comportement par défaut), la fonction ne renverra que les
#   carrés où il a été détecté un visage
# mettre return_pos à 0 pour désactiver ce comportement et obtenir toutes les
#   carrés analysés
def fenetre_glissante(clf, scoreValidation, img, ratio, newSize, pas_hor, pas_vert, tailleDescripteur, return_pos=1):
    img = color.rgb2gray(img)

    # on détermine les bordures de l'image
    limite_y, limite_x = img.shape

    # calcul du nombre de fenetres glissantes
    dim_x = int((limite_x - newSize) / pas_hor) + 1
    dim_y = int((limite_y - newSize) / pas_vert) + 1
    data = np.zeros(((dim_x + 1) * (dim_y + 1) -1, 5))

    # print("nb pas_x", dim_x, "nb pas_y", dim_y)

    indice = 0
    # on parcourt l'ensemble de l'image selon x, y
    for i in range(0, dim_x):
        for j in range(0, dim_y):
            x_tmp = i*pas_hor
            y_tmp = j*pas_vert
            img_tmp = img[y_tmp:y_tmp + newSize, x_tmp:x_tmp + newSize]
            img_tmp = filtreLineaire(img_tmp)
            data[indice] = [clf.decision_function(img_tmp), x_tmp, y_tmp, newSize, newSize]
            #data[indice] = [clf.predict_proba(img_tmp)[0][1], x_tmp, y_tmp, newSize, newSize]
            indice += 1

    # derniere ligne selon y (x = cst)
    for i in range(0, dim_y):
        x_tmp = limite_x - newSize
        y_tmp = i * pas_vert
        img_tmp = img[y_tmp:y_tmp + newSize, x_tmp:x_tmp + newSize]
        img_tmp = filtreLineaire(img_tmp)
        
        data[indice] = [clf.decision_function(img_tmp), x_tmp, y_tmp, newSize, newSize]
        #data[indice] = [clf.predict_proba(img_tmp)[0][1], x_tmp, y_tmp, newSize, newSize]

        indice += 1

    # derniere ligne selon x (y = cst)
    for i in range(0, dim_x):
        x_tmp = i * pas_hor
        y_tmp = limite_y - newSize
        img_tmp = img[y_tmp:y_tmp + newSize, x_tmp:x_tmp + newSize]
        img_tmp = filtreLineaire(img_tmp)

        data[indice] = [clf.decision_function(img_tmp), x_tmp, y_tmp, newSize, newSize]
        #data[indice] = [clf.predict_proba(img_tmp)[0][1], x_tmp, y_tmp, newSize, newSize]

        indice += 1

    #On remetà la taille de l'image d'origine
    data[:,1:] /= ratio
    if return_pos:
        return data[data[:, 0] >= scoreValidation]
    else:
        return data


# affiche une image avec le rectangle associé
def afficher_fenetre_gliss(img, data_fenetre, scoremin, only_pos=0,animated=0):
    detections = 0
    if animated==0:
        # Créer la figure et les axes
        fig,ax = plt.subplots(1)
        # Afficher l'image
        ax.imshow(img, cmap='gray')
        # Créer le rectangle
    for i in range(1, np.size(data_fenetre, 0) + 1):
        if animated==1:
            # Créer la figure et les axes
            fig,ax = plt.subplots(1)
            # Afficher l'image
            ax.imshow(img, cmap='gray')
            # Créer le rectangle
        score, xcorner, ycorner, width, height = data_fenetre[i-1]
        if (score >= scoremin) or (not only_pos):
            if score >= scoremin:
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
def suppressionNonMaximas(data, facteur=0.1):
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
                # Si leur recouvrement est supérieur à 50% on considère que
                # c'est une boite redondante et on la supprime donc
                if recouvrement(xi,yi,wi,hi,xj,yj,wj,hj) > facteur:
                    data[j][0] = 0
        # On passe à la boîte suivante
        i+=1
    # Retourne les boites qui n'ont pas été éliminées (score != 0)
    return data[data[:,0] != 0]

# lance la fenêtre glissante sur différentes échelles, retourne les meilleures
# fenêtres détectées, après nettoyage des doublons par suppressionNonMaximas()
def fenetre_glissante_multiechelle(clf, scoreValidation, img, newSize,
                                   tailleDescripteur, animated=0,
                                   return_pos=1):
    data = np.zeros((10000, 5)) # Max 100 fenêtres
    cursor = 0
    ratio = newSize/min(img.shape)
    while ratio < 1:
        # print("Fenêtre glissante",round(ratio*100),"%")
        data_f = fenetre_glissante(clf, scoreValidation, rescale(img, ratio, mode='reflect'),
                                   ratio, newSize, newSize//4,newSize//4, tailleDescripteur,
                                   return_pos)
        if animated==1:
            afficher_fenetre_gliss(img, data_f, 0, only_pos=0,animated=1)
        for i in range(len(data_f)):
            data[i+cursor] = data_f[i]
        cursor += len(data_f)
        ratio *= 1.3
    return data

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Faux positifs et calcul des résultats

# calcul des résultas sur les données, selon le formatage donné
# numeroImage X Y L H score
def calculResultats(clf, scoreValidation, path, newSize, tailleDescripteur,
                    debut=0, fin=-1, etat=0):
    if fin == -1:
        fin = len(os.listdir(path))

    data_res = np.zeros((100000, 6)) # initialisation de l'array des résultats
    cursor = 0 # indice où sera inséré le prochain résultat
    for i in range(debut, fin):
        if etat:
            pct = round(100*((i-debut)/(fin-debut)))
            print("\r" + str(pct//2*"-"+"{}% ({})".format(pct, len(data_res[data_res[:,0]!=0]))), end="\r")

        # lecture et détection des visages dans l'ième image
        img = np.array(io.imread(path +"%04d"%(i+1)+".jpg", as_grey=True))
        data_f = fenetre_glissante_multiechelle(clf, scoreValidation, img,
                                                newSize, tailleDescripteur)
        # écriture des résultats dans l'array de résultats
        for j in range(len(data_f)):
            data_res[j+cursor, 0] = i+1
            data_res[j+cursor, 1:5] = data_f[j, 1:]
            data_res[j+cursor, 5] = data_f[j, 0]
            cursor += 1
    if etat:
        print()
    # renvoie les résultats non nuls
    return data_res[data_res[:,0] != 0]

# mets les datas sous la forme
# N x y w h 1 si vrai positif
# N x y w h -1 si faux positif
def calculResultatsTrain(clf, data, data_res):
    for i in range(len(data_res)):
        [n, x, y, w, h] = data_res[i][0:5]
        
        # récupération des coordonnes du visage correspondant à l'image n
        [xn, yn, wn, hn] = data[int(n) - 1][1:5]
        
        if recouvrement(xn, yn, wn, hn, x, y, w, h) < 0.3:
            data_res[i][5] = -1
        else:
            data_res[i][5] = 1

    return data_res

# data_res =  N x y w h (-)1 si vrai (faux) positif
def affichAnalyseResultat(data_res, data):
    rappel = len(data_res[data_res[:,5]!=-1]) / len(data)
    precision =  len(data_res[data_res[:,5]!=-1]) / len(data_res)
    
    return rappel, precision

# data_calc = # numeroImage X Y L H score
def courbePrecisionRappel(data_calc, data, show = 0):
    # tri des datas dans par ordre décroissant de score
    data_res_sort = data_calc[data_calc[:,5].argsort()]

    data_res = calculResultatsTrain(0, data, data_res_sort)
    
    # on rajoute les données une à unes et on calcule de nouveau précision et
    # rappel
    rappel_precision = np.zeros((len(data_res) - 1, 2))
    for i in range(1, len(data_res)):
        rappel_precision[i - 1] = affichAnalyseResultat(data_res[len(data_res)-i:],
                                                        data)
    
    if show == 1:
        plt.plot(rappel_precision[:,0], rappel_precision[:,1])
        plt.ylabel('Précision')
        plt.xlabel('Rappel')
        plt.show()
        
    return rappel_precision
    

def fauxPositifs(clf, pathTrain, data, newSize, tailleDescripteur, scoreValidation,
                 debut=0, fin=-1, etat=0):
    data_res = calculResultats(clf, scoreValidation, pathTrain, newSize,
                               tailleDescripteur, debut, fin, etat)
    
    rappel, precision = affichAnalyseResultat(data_res, data)
    
    print("rappel : ", rappel, ", precision : ", precision)
    
    data_res = calculResultatsTrain(clf, data, data_res)
    
    return data_res[data_res[:,5]==-1]
