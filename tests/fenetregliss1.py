#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 19 12:10:00 2017

@author: marie
"""

# à lancer après ./projetComplet.py

import pyfacedetect.image as libimg

img = np.array(Image.open(pathTrain +"%04d"%(14)+".jpg"), dtype=np.uint8)
data_f = libimg.fenetre_glissante(clf, img, 30, 30, 15,20)
libimg.afficher_fenetre_gliss(img, data_f, pathTrain)
