# Entrainement d'un CNN pour une tâche de reconnaissance d'animaux

## Prétraitement des images
D'abord, le fichier findsize.py permet de trouver la gamme des dimensions des images dans les données afin de savoir quel padding est nécessaire.
Ensuite, le fichier preprocess.py effectue le prétraitement approprié des données (les tranformant en images 256x256 en employant le padding.

## Entrainement et évaluation du modèle
L'entrainement du modèle se fait sur les données prétraitées grâce au fichier 1_Modele.py, et l'évaluation se fait dans le fichier 2_Evaluation.py. Ces fichiers sont bien commentés et incluent les détails d'entrainement ainsi qu'une description des données et de l'architecture du modèle.
