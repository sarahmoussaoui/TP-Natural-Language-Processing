<!-- @format -->

# 📚 Séance N° 2 - Modèles de Sémantique Vectorielle et Skip-Gram (Word2Vec)

## 1. Modèles de Sémantique Vectorielle

La sémantique vectorielle cherche à représenter la signification des mots dans un espace multidimensionnel.

### A. Modèles Fondamentaux

- **Modèles basés sur la Connotation** : La signification est représentée par les connotations du mot.
- **Modèles basés sur la Distribution** : La signification est dérivée de la distribution. **L'Hypothèse Distributionnelle** stipule que les mots apparaissant dans des contextes similaires ont des significations proches.

### B. Familles de Modèles Distributionnels

| Famille de Modèles | Caractéristiques                                                                                              | Vecteurs                                                                                                                                    | Exemples             |
| ------------------ | ------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------- | -------------------- |
| **Co-occurrence**  | Représentent la signification à partir des fréquences des mots voisins (co-occurrence).                       | **Longs et Creux (sparse)**. Dimensions : documents (matrice Terme-Document, pondérée TF-IDF) ou mots (matrice Terme-Terme, pondérée PPMI). | TF-IDF, PPMI         |
| **Apprentissage**  | Apprennent directement des représentations vectorielles (Embeddings) qui capturent les relations sémantiques. | **Courts et Denses**. Dimension $d$ (50 à 1000) est beaucoup plus petite que la taille du vocabulaire $V$.                                  | Word2vec (Skip-Gram) |

### C. Avantages des Embeddings Denses (vs. Creux)

Les Word Embeddings sont des vecteurs courts et denses qui sont généralement plus efficaces pour les tâches de TALN :

- **Efficacité** : Ils contiennent des valeurs réelles, y compris négatives.
- **Généralisation** : L'espace des paramètres est réduit (ex: 300 dimensions vs. 50 000 dimensions pour les vecteurs creux), ce qui facilite la généralisation et aide à éviter le surapprentissage.
- **Sémantique** : Les vecteurs denses peuvent mieux capturer les relations de synonymie, car les dimensions associées aux synonymes ne sont pas distinctes.

## 2. Skip-Gram avec Échantillonnage Négatif (SGNS)

Skip-Gram (souvent appelé SGNS) est un algorithme de word2vec utilisé pour générer des embeddings de mots statiques (un embedding fixe par mot).

### A. Principe

- Skip-Gram reformule l'apprentissage de la sémantique comme une tâche de **classification binaire** : prédire si un mot $\boldsymbol{c}$ a une forte probabilité d'apparaître près du mot cible $\boldsymbol{w}$.
- **Objectif Principal** : Le véritable objectif n'est pas la classification, mais d'apprendre les poids du classifieur, car ces poids constituent directement les embeddings de mots.

### B. Génération des Données d'Entraînement

Le modèle utilise le texte brut pour générer des données auto-supervisées (sans étiquetage manuel).

- **Exemples Positifs (Label 1)** : Une paire $(\boldsymbol{w}, \boldsymbol{c}_{\boldsymbol{pos}})$ est positive si $\boldsymbol{c}_{\boldsymbol{pos}}$ est un mot de contexte réel trouvé dans la fenêtre de voisinage du mot cible $\boldsymbol{w}$.
- **Échantillonnage Négatif** : Pour chaque exemple positif, le modèle génère $k$ échantillons négatifs $(\boldsymbol{w}, \boldsymbol{c}_{\boldsymbol{neg}})$.
  - Le mot de bruit $\boldsymbol{c}_{\boldsymbol{neg}}$ est choisi aléatoirement dans le vocabulaire (à l'exception de $\boldsymbol{w}$ et $\boldsymbol{c}_{\boldsymbol{pos}}$).
  - Les mots de bruit sont échantillonnés selon une distribution Unigramme pondérée $P_\alpha(\boldsymbol{w})$ avec $\boldsymbol{\alpha = 0.75}$. Cette pondération augmente légèrement la probabilité de sélectionner des mots rares comme mots de bruit, améliorant les performances.

### C. Calcul de Probabilité et Embeddings

Le modèle utilise la similarité entre les embeddings (produit scalaire) pour estimer la probabilité de co-occurrence.

- **Similarité** : $\text{similarity}(\boldsymbol{w}, \boldsymbol{c}) \approx \boldsymbol{c} \cdot \boldsymbol{w}$
- **Probabilité** : La fonction Sigmoïde ($\sigma$) est appliquée au produit scalaire pour obtenir une probabilité :

$$
P^+(\boldsymbol{w}, \boldsymbol{c}) = \sigma(\boldsymbol{c} \cdot \boldsymbol{w}) = \frac{1}{1 + \exp(-\boldsymbol{c} \cdot \boldsymbol{w})}
$$

### D. Paramètres du Modèle

Skip-Gram maintient deux types d'embeddings pour chaque mot $i$ :

- Un vecteur d'embedding cible $\boldsymbol{w}_i$ (matrice $\boldsymbol{W}$).
- Un vecteur d'embedding contexte $\boldsymbol{c}_i$ (matrice $\boldsymbol{C}$).
- Après l'entraînement, le mot $i$ est représenté uniquement par son embedding cible $\boldsymbol{w}_i$.

### E. Apprentissage (Descente de Gradient)

L'apprentissage se fait par **Descente de Gradient Stochastique (SGD)** en minimisant la fonction de perte $L_{CE}$. L'objectif est :

- Maximiser la similarité entre le vecteur du mot cible $\boldsymbol{w}$ et les vecteurs de contexte réels $\boldsymbol{c}_{\boldsymbol{pos}}$ (les rapprocher).
- Minimiser la similarité entre le vecteur du mot cible $\boldsymbol{w}$ et les $k$ mots de bruit $\boldsymbol{c}_{\boldsymbol{neg}}$ (les éloigner).

La fonction de perte pour un exemple positif et ses $k$ exemples négatifs est :

$$
L_{CE} = - \log P^+(\boldsymbol{w}, \boldsymbol{c}_{\boldsymbol{pos}}) - \sum_{i=1}^k \log P^-(\boldsymbol{w}, \boldsymbol{c}_{\boldsymbol{neg}_i})
$$

(où $P^-(\boldsymbol{w}, \boldsymbol{c}) = 1 - P^+(\boldsymbol{w}, \boldsymbol{c})$)

**La taille de la fenêtre contextuelle $L$ est un paramètre important qui doit être ajusté pour optimiser les performances.**
