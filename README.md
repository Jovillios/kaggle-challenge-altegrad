# Kaggle Challenge: Llama 3.1 8B Finetuning

Ce projet contient des scripts pour le fine-tuning du modèle Llama 3.1 8B sur un dataset spécifique pour une compétition Kaggle. Le projet inclut des scripts pour le prétraitement des données, l'entraînement du modèle et l'inférence.

## Structure du projet

- inference.py : Script pour générer les prédictions sur un jeu de données de test.
- install.sh : Script pour télécharger et extraire les données de la compétition Kaggle.
- preprocess.py : Script pour prétraiter les données.
- requirements.txt : Fichier contenant les dépendances du projet.
- train.py : Script pour entraîner le modèle (SFT)
- to_csv.py : Script pour convertir les prédictions en fichier CSV pour soumission sur Kaggle.

## Installation

1. Clonez le dépôt :

   ```sh
   git clone <URL_DU_DEPOT>
   cd <NOM_DU_DEPOT>
   ```

2. Installez les dépendances :

   ```sh
   pip install -r requirements.txt
   ```

3. Téléchargez et extrayez les données de la compétition Kaggle :
   ```sh
   bash install.sh
   ```

## Utilisation

### Prétraitement des données

Pour prétraiter les données, exécutez le script [preprocess.py](preprocess.py) :

```sh
python preprocess.py

```

### Entraînement du modèle

Pour entraîner le modèle, exécutez le script [train.py](train.py) :

```sh
python train.py
```

### Génération des prédictions

Pour générer les prédictions sur un jeu de données de test, exécutez le script [inference.py](inference.py) :

```sh
python inference.py
```

### Conversion des prédictions en fichier CSV

Pour convertir les prédictions en fichier CSV pour soumission sur Kaggle, exécutez le script [to_csv.py](to_csv.py) :

```sh
python to_csv.py
```
