# Entrainement d'un CNN pour une tâche de reconnaissance d'animaux

## Description des fichiers
D'abord, le fichier **findsize.py** permet de trouver la gamme des dimensions des images dans les données afin de savoir quel padding est nécessaire.
Ensuite, le fichier **preprocess.py** effectue le prétraitement approprié des données (les tranformant en images 256x256 en employant le padding.
L'entrainement du modèle se fait sur les données prétraitées grâce au fichier **1_Modele.py**, et l'évaluation se fait dans le fichier **2_Evaluation.py**.

Le fichier **Philip_Voinea_TP2.pdf** contient le raport complèt du projet, incluant l'évolution, l'évaluation du modèle, la justification de l'architecture et plus.

# Construction d’un modèle de classification d’images d’animaux
