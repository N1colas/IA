#APPROXIMATION DE FONCTION AVEC UN RESEAU DE NEURONES
# Cédric Vasseur, www.beepmaster.com
#
#Python 64 bits windows version 3.8.6rc1
#pip install tensorflow==2.3.0
#pip install keras==2.4.3
#pip install matplotlib==3.3.2

#Nous allons "apprendre" à un réseau de neurones
#à imiter la fonction "Cosinus" pour cela nous allons :
#
#Prérequis : Python, Tensorflow, Numpy, Keras, Matplotlib
#
#0/ Voir https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/
#1/ Importer les packages nécessaire pour notre projet
#2/ Créer une jeu de données d'entraînement à partir de notre fonction cosinus
#3/ Initialiser notre modèle
#4/ Entraîner notre modèle
#5/ Afficher le résultat sous forme de Graphe
#
#Voir aussi :
#
#George Cybenko (1989)
# “Approximations by superpositions of sigmoidal functions”,  Mathematics of Control, Signals, and Systems
#Kurt Hornik (1991)
# “Approximation Capabilities of Multilayer Feedforward Networks”,  Neural Networks

#on force l'utilisation du CPU a la place du GPU pour Tensorflow
import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"  

#1/IMPORT


#Import de Math (ne pas confondre avec Matplotlib)
#Pour les fonctions mathématiques de base de Python (tel que math.pi ...)
import math

#Import de Numpy
#Pour faciliter la manipulation de tableaux de données
import numpy as np

#Import de Keras
#Pour faciliter l'écriture en quelques lignes notre réseau de neurones
from keras.layers import Dense
from keras.models import Sequential
from keras.layers import Activation

#Import de matpoltlib
#Pour afficher le résultat sous la forme d'un graphique
import matplotlib.pyplot as plt


#2/DONNEES D'ENTRAINEMENT

#On créé un tableau de données avec pour limites -2*math.pi et math.pi*2 et intervalle de 0.1
x = np.arange(-2*math.pi, math.pi*2, .1)
#On calcule les valeurs de y sur cet intervalle avec pour valeur cos(x) ...
y = (np.cos(x)+1)/2

#3/ INITIALISATION DU MODELE

model = Sequential([
    Dense(10, input_shape=(1,)),    #couche cachée de 10 neurones à une dimension
    Activation('sigmoid'),          #fonction d'activation : sigmoid
    Dense(1)                        #une entrée
])
#Compilation du modèle
#Calcul de la fonction de coût avec l'erreur quatratique moyenne
#Optimiseur : avec AdamBoost (autre optimiseur possibles : SVG...)
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_squared_error'])


#4/ ENTRAINEMENT

#Entrées / paramètres : x et y
#Epoch : nombre de générations / de fois où le modèle sera entraîné par le jeu d'entraînement
#        plus cette valeur est grande plus la prédiction sera précise
#Batch_size : taille du batch
#Verbose : Mode verbeux / la valeur 2 permet d'afficher une ligne par génération
model.fit(x, y, epochs=3000, batch_size=8, verbose=2) #Verbose 2 permet d'afficher une ligne par generation

#5/ AFFICHER LE RESULTAT

predictions = model.predict(x)

#Affiche une ligne bleue (données réelles), et une ligne rouge discontinue la prédiction
plt.plot(x, y, 'b', x, predictions, 'r--') #b = ligne bleu / r-- = ligne discontinue rouge
plt.ylabel('y') #Etiquette de l'axe y
plt.xlabel('x') #Etiquette de l'axe x
plt.show() #affiche le résultat

#A/ Testez d'autres valeurs pour Epoq : que constatez-vous ?
#B/ Testez d'autres optimiseur : SVG à la place d'adam par exemple, que constatez-vous ?
#C/ Quelles critiques pouvez vous faire sur le jeu de données utilisé ?
#D/ Testez différents hyperparamètres sur notre modèles : nombre de couches,
#   type de fonction d'activation (relu par exemple) ...
#E/ Changez les couleurs et étiquettes du graphique.






