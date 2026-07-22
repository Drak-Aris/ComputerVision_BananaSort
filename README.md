# BananaVision — Tri automatisé de bananes par vision par ordinateur

Système d'aide à la décision pour le **tri et le routage de bananes Cavendish** sur une ligne de conditionnement (convoyeur). Le projet combine **deep learning**, **apprentissage automatique classique** et **processus décisionnels markoviens (MDP)** afin de classifier chaque fruit, estimer la confiance des prédictions et recommander une action commerciale optimale.

Contexte métier : **Plantations du Haut-Penja (PHP)**, Cameroun — projet académique **UCAC-ICAM 2026**.

---

## Sommaire

- [Problématique](#problématique)
- [Architecture globale](#architecture-globale)
- [Les 6 classes de classification](#les-6-classes-de-classification)
- [Structure du dépôt](#structure-du-dépôt)
- [Prérequis et installation](#prérequis-et-installation)
- [Pipeline de traitement (production)](#pipeline-de-traitement-production)
- [Entraînement des modules](#entraînement-des-modules)
- [API REST (FastAPI)](#api-rest-fastapi)
- [Actions de décision (Module C)](#actions-de-décision-module-c)
- [Journalisation et tableau de bord](#journalisation-et-tableau-de-bord)
- [Contraintes industrielles](#contraintes-industrielles)
- [Technologies utilisées](#technologies-utilisées)

---

## Problématique

Sur une ligne de tri, chaque banane doit être orientée vers l'une des filières suivantes :

| Filière | Description |
|---------|-------------|
| **Export Catégorie I** | Fruit vert sain, conforme à la norme UE 1333/2011 — marché européen premium |
| **Marché local Catégorie II** | Fruit mûr sain — écoulement Douala / Mungo |
| **Transformation industrielle** | Fruit trop mûr sain — farine, jus, sous-produits |
| **Suspension / contrôle manuel** | Ambiguïté, fruit malade ou risque phytosanitaire |

L'enjeu économique est direct : un **faux négatif** (exporter un mauvais fruit) expose à des rejets douaniers ; un **faux positif** (rejeter un bon fruit) représente une perte de revenu. Le système vise un temps d'inférence **< 200 ms** par fruit pour rester compatible avec le débit du convoyeur.

---

## Architecture globale

Le projet est organisé en **trois modules** qui s'enchaînent dans un pipeline unifié :

```
┌─────────────┐     ┌──────────────────────┐     ┌─────────────────┐
│   Image     │────▶│  Module A — Deep     │────▶│  Classe CNN +   │
│  (caméra)   │     │  Learning (CNN)      │     │  probabilités   │
└─────────────┘     └──────────────────────┘     └────────┬────────┘
                                                          │
                        ┌──────────────────────┐          │
                        │  Module B — ML         │◀─────────┘
                        │  (K-Means sur          │   embedding CNN
                        │   embeddings)          │
                        └────────┬───────────────┘
                                 │ cluster_id, filière, confiance
                                 ▼
                        ┌──────────────────────┐
                        │  Module C — MDP      │
                        │  (Value Iteration)   │
                        └────────┬─────────────┘
                                 │
                                 ▼
                        Action optimale
                  (export / local / transfo / suspend)
```

### Module A — Classification par CNN (`DL/`)

- **Modèle** : EfficientNet-B0 (transfer learning ImageNet), tête de classification à 6 classes.
- **Entrée** : image RGB 224×224, normalisée (moyenne/écart-type ImageNet).
- **Sorties** :
  - Classe prédite et score de confiance (softmax).
  - Vecteur d'embedding (1280 dimensions) extrait de l'avant-dernière couche.
  - Probabilités binaires export / rejet (`vert_sain` = export, les autres = rejet).
- **Entraînement** : early stopping, pondération des classes déséquilibrées, courbe ROC binaire export/rejet, analyse des coûts métier (FP/FN).

### Module B — Clustering non supervisé (`ML/unsupervised/`)

- **Entrée** : embeddings du Module A (`embeddings_train.csv`).
- **Méthode** : StandardScaler + K-Means (k = 6), sélection de k par coude et score de silhouette (`k_mean.py`).
- **Sorties** :
  - `cluster_id` (0–5).
  - Mapping cluster → **filière métier** (`malade`, `vert_sain`, `mure_sain`, `tropmure_sain`) via `cluster_to_classe.joblib`.
  - Score de confiance du cluster (distance au centroïde vs. second centroïde le plus proche).
- **Complément** : extraction de features couleur/forme OpenCV (`ML/feature/featuring.py`) et classifieur Naïve Bayes (`ML/supervised/`) pour une approche alternative export/rejet.

### Module C — Décision MDP (`moduleC/`)

- **État** : combinaison de 20 états possibles construits à partir de :
  - **Groupe** (4) : malade, vert_sain, mure_sain, tropmure_sain — dérivé du cluster.
  - **Confiance** (3) : faible, moyen, fort — score global `0.6 × binary_conf + 0.4 × cluster_conf`.
  - **Alerte** (2) : 0 ou 1 — déclenchée si confiance CNN ou cluster insuffisante.
- **Actions** (4) : `export_cat1`, `local_cat2`, `transformation`, `suspend`.
- **Algorithme** : Value Iteration (γ = 0.95) avec matrices de récompense (FCFA/fruit) et de transition calibrées sur les données économiques PHP (prix export, marché local, transformation, coûts de rejet).
- **Politique** : exportée dans `moduleC/module_c/politique_optimale.json`, chargée en O(1) par `MDPEngine`.

---

## Les 6 classes de classification

Chaque image est étiquetée selon la **maturation** et l'**état sanitaire** :

| Classe | Maturité | Santé | Filière cible typique |
|--------|----------|-------|------------------------|
| `vert_sain` | Vert | Sain | Export |
| `vert_malade` | Vert | Malade | Rejet / contrôle |
| `mure_sain` | Mûr | Sain | Marché local |
| `mure_malade` | Mûr | Malade | Rejet / contrôle |
| `tropmure_sain` | Trop mûr | Sain | Transformation |
| `tropmure_malade` | Trop mûr | Malade | Rejet / contrôle |

Le dataset contient environ **5 100 images d'entraînement**, **660 de validation** et **505 de test**, réparties dans `dataset/train`, `dataset/valid` et `dataset/test`.

---

## Structure du dépôt

```
ComputerVision_BananaSort/
├── pipeline.py              # API FastAPI — point d'entrée production
├── log_results.py           # Journalisation CSV des prédictions
├── requirements.txt         # Dépendances Python
│
├── dataset/
│   ├── data_processing.py   # Nettoyage, redimensionnement, DataLoaders PyTorch
│   ├── train/               # 6 sous-dossiers par classe
│   ├── valid/
│   └── test/
│
├── DL/
│   ├── cnn_efficientnet-b0.py   # Entraînement et évaluation du CNN
│   ├── best_model.pth             # Poids du modèle entraîné
│   └── embeddings_train.csv       # Embeddings extraits pour le Module B
│
├── ML/
│   ├── feature/featuring.py           # Features couleur/contour (OpenCV)
│   ├── supervised/naivesbayes_classification.py
│   └── unsupervised/
│       ├── k_mean.py                  # Choix optimal de k
│       ├── clustering.py              # K-Means + mapping filières
│       ├── scaler.joblib
│       ├── kmeans_model.joblib
│       └── cluster_to_classe.joblib
│
└── moduleC/
    ├── state_builder.py       # Construction des 20 états MDP
    ├── train_mdp.py           # Entraînement VI/PI + export politique
    ├── mdp_engine.py          # Moteur de décision (lookup O(1))
    ├── module_c/politique_optimale.json
    └── figures/               # Graphiques de convergence, V*(s), sensibilité γ
```

---

## Prérequis et installation

- **Python** 3.10+
- **GPU CUDA** recommandé (PyTorch cu118 inclus dans `requirements.txt`)
- Environnement virtuel conseillé

```bash
# Cloner le dépôt
git clone <url-du-repo>
cd ComputerVision_BananaSort

# Créer et activer un environnement virtuel
python -m venv .envbanana
source .envbanana/bin/activate   # Linux / macOS
# .envbanana\Scripts\activate    # Windows

# Installer les dépendances
pip install -r requirements.txt
```

> **Note** : les modèles pré-entraînés (`best_model.pth`, fichiers `.joblib`, `politique_optimale.json`) doivent être présents aux chemins attendus par `pipeline.py`. Relancer les scripts d'entraînement si nécessaire (voir ci-dessous).

---

## Pipeline de traitement (production)

Le fichier `pipeline.py` orchestre l'ensemble du flux pour une image entrante :

1. **Chargement** des ressources au démarrage (CNN, scaler, K-Means, politique MDP).
2. **Inférence CNN** → classe, confiance, probabilités, embedding.
3. **Clustering** → `cluster_id`, filière métier, confiance cluster.
4. **Construction d'état MDP** → `state_id` parmi 20 états.
5. **Décision optimale** → action recommandée avec justification et valeur V*.
6. **Journalisation** dans `banana_vision_log.csv`.

Lancer l'API :

```bash
uvicorn pipeline:app --host 0.0.0.0 --port 8000 --reload
```

Documentation interactive : [http://localhost:8000/docs](http://localhost:8000/docs)

---

## Entraînement des modules

Exécuter les scripts **dans l'ordre** suivant (depuis la racine du projet, sauf indication) :

| Étape | Commande | Sortie principale |
|-------|----------|-------------------|
| 1. Préparation dataset | *(automatique via `get_dataset_loaders`)* | Images 224×224 |
| 2. CNN | `python DL/cnn_efficientnet-b0.py` | `DL/best_model.pth`, `embeddings_train.csv` |
| 3. Choix de k | `python ML/unsupervised/k_mean.py` | Graphiques coude / silhouette |
| 4. Clustering | `python ML/unsupervised/clustering.py` | `scaler.joblib`, `kmeans_model.joblib`, `cluster_to_classe.joblib` |
| 5. MDP | `python moduleC/train_mdp.py` | `politique_optimale.json`, figures |

*(Optionnel)* Features OpenCV + Naïve Bayes :

```bash
python ML/feature/featuring.py
python ML/supervised/naivesbayes_classification.py
```

---

## API REST (FastAPI)

### `POST /predict`

Envoie une image et reçoit la prédiction complète du pipeline.

**Requête** : `multipart/form-data` avec champ `file` (image JPEG/PNG).

**Réponse** (extrait) :

```json
{
  "status": "success",
  "data": {
    "classe_cnn": "vert_sain",
    "confiance_cnn": 0.94,
    "cluster_id": 5,
    "filiere": "vert_sain",
    "state_id": 12,
    "state_libelle": "Fruit vert sain — Export — confiance fort — sans alerte",
    "action": "export_cat1",
    "action_libelle": "✅ Export Catégorie I (marché européen)",
    "V_star": 788.45,
    "justification": "...",
    "temps_ms": 45.2
  }
}
```

### `GET /dashboard`

Retourne des statistiques agrégées depuis `banana_vision_log.csv` :

- Nombre total d'analyses
- Distribution des actions et filières
- Série temporelle (analyses par minute)
- Confiance et temps d'inférence moyens
- 10 dernières analyses

---

## Actions de décision (Module C)

| Code | Action | Libellé |
|------|--------|---------|
| A1 | `export_cat1` | Export Catégorie I (marché européen) |
| A2 | `local_cat2` | Marché local Catégorie II (Douala / Mungo) |
| A3 | `transformation` | Transformation industrielle (farine / jus) |
| A4 | `suspend` | Suspension — contrôle manuel requis |

Les récompenses du MDP sont exprimées en **FCFA par fruit**, à partir de paramètres métier documentés dans `train_mdp.py` (prix export ~400 FCFA/kg, poids moyen 150 g, coûts de rejet douanier, etc.).

---

## Journalisation et tableau de bord

Chaque appel à `/predict` est enregistré dans `banana_vision_log.csv` avec :

`timestamp`, `image`, `classe_cnn`, `confiance_cnn`, `cluster_id`, `filiere`, `state_id`, `action`, `action_libelle`, `v_star`, `temps_ms`

Le endpoint `/dashboard` exploite ce fichier pour alimenter un frontend de suivi (graphiques camembert, courbes temporelles, histogrammes de latence).

---

## Contraintes industrielles

- **Latence** : le CNN est benchmarké pour rester sous **200 ms/image** (seuil convoyeur PHP).
- **Robustesse** : les états à faible confiance ou avec alerte déclenchent preferentiellement `suspend`.
- **Réglementation** : l'export (`export_cat1`) est masqué pour les fruits malades en alerte (règle phytosanitaire).
- **GPU** : CUDA utilisé automatiquement si disponible ; repli CPU sinon.

---

## Technologies utilisées

| Domaine | Bibliothèques |
|---------|----------------|
| Deep Learning | PyTorch, torchvision (EfficientNet-B0) |
| Vision / features | OpenCV, Pillow |
| Machine Learning | scikit-learn (K-Means, Naïve Bayes, PCA) |
| API | FastAPI, uvicorn, python-multipart |
| Données | pandas, numpy, joblib, pyarrow |
| Visualisation | matplotlib, seaborn |
| Décision | NumPy (Value Iteration, Policy Iteration) |

---

## Licence

Projet académique — UCAC-ICAM 2026, en partenariat avec Plantations du Haut-Penja (PHP).
