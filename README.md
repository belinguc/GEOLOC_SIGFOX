# GEOLOC_SIGFOX

## matrix.py

Ce fichier permet de générer différentes matrices pour l'apprentissage à partir du jeu de données fourni.
Nous avons testé à l'intérieur du fichier différents préprocessing et feature engineering puis nous avons évalué les performances par la suite.

## stats_descriptives.ipynb

Il s'agit du notebook utilisé pour le travail exploratoire sur les données. Il comporte aussi les visualisations que nous avons effectuées pour mieux se rendre compte des données.

## split.ipynb

Dans ce notebook nous avons effectué un split sur les données de train afin d'avoir un jeu de données de validation pour évaluer le score de notre modèle. Nous avons fait attention à prendre des objets différents entre le training set et le validation set.

## verify_split.ipynb

Nous avons ici vérifié que les distributions étaient similaires entre les jeux de train et de validation.

## tools.py

Le fichier regroupe des fonctions utilitaires pour afficher les scores des modèles...

## main.ipynb

Il s'agit de la partie qui récupère les csv sortis par matrix.py et applique les modèles.

## my_map.html

Il s'agit de la dernière figure de visualisation de données.
