import streamlit as st
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import os
import re
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# Configuration de la page
st.set_page_config(
    page_title="SkipGram - Partie 2",
    page_icon="🔍",
    layout="wide"
)

# Titre de l'application
st.title("🔍 SkipGram avec Negative Sampling - Partie 2")
st.markdown("**TP N°8 - Word Embedding (Part 2) - TALN - Master 2 SII**")
st.markdown("---")

# ===============================================
# DÉFINITION EXACTE DU MODÈLE COMME DANS PARTIE 1
# ===============================================
class SkipGramModel(nn.Module):
    """Même architecture que dans la partie 1"""
    def __init__(self, vocab_size, embedding_dim):
        super(SkipGramModel, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        
        # Embeddings pour les mots centre
        self.center_embeddings = nn.Embedding(vocab_size, embedding_dim)
        # Embeddings pour les mots contexte
        self.context_embeddings = nn.Embedding(vocab_size, embedding_dim)
        
    def forward(self, center_words, context_words):
        center_embed = self.center_embeddings(center_words)
        context_embed = self.context_embeddings(context_words)
        # Produit scalaire (comme demandé dans le TP)
        scores = torch.sum(center_embed * context_embed, dim=1)
        return scores
    
    def get_word_vector(self, word_idx):
        """Récupère le vecteur d'un mot (embedding centre)"""
        with torch.no_grad():
            return self.center_embeddings(torch.LongTensor([word_idx])).squeeze(0).numpy()

# ===============================================
# FONCTIONS POUR CHARGER LE MODÈLE
# ===============================================
@st.cache_resource
def charger_modele_skipgram(chemin_modele: str):
    """
    Charge le modèle SkipGram exactement comme sauvegardé dans la partie 1
    """
    try:
        # Vérifier si le fichier existe
        if not os.path.exists(chemin_modele):
            st.error(f"❌ Fichier {chemin_modele} non trouvé!")
            return None
        
        # Charger le checkpoint
        checkpoint = torch.load(chemin_modele, map_location=torch.device('cpu'))
        
        # Extraire les paramètres
        vocab_size = checkpoint.get('vocab_size')
        embedding_dim = checkpoint.get('embedding_dim')
        word_to_idx = checkpoint.get('word_to_idx', {})
        idx_to_word = checkpoint.get('idx_to_word', {})
        
        if vocab_size is None or embedding_dim is None:
            st.error("❌ Le fichier modèle ne contient pas vocab_size ou embedding_dim")
            return None
        
        # Créer le modèle avec les bons paramètres
        modele = SkipGramModel(vocab_size, embedding_dim)
        
        # Charger les poids
        if 'model_state_dict' in checkpoint:
            modele.load_state_dict(checkpoint['model_state_dict'])
        else:
            # Essayer de charger directement
            modele.load_state_dict(checkpoint)
        
        modele.eval()  # Mode évaluation
        
        st.success(f"✅ Modèle chargé avec succès!")
        st.success(f"   - Vocab size: {vocab_size}")
        st.success(f"   - Embedding dim: {embedding_dim}")
        st.success(f"   - Taille vocabulaire: {len(word_to_idx)} mots")
        
        return {
            'model': modele,
            'vocab_size': vocab_size,
            'embedding_dim': embedding_dim,
            'word_to_idx': word_to_idx,
            'idx_to_word': idx_to_word
        }
        
    except Exception as e:
        st.error(f"❌ Erreur lors du chargement: {str(e)}")
        return None

# ===============================================
# FONCTIONS POUR LES SIMILARITÉS (CORRIGÉ - COSINUS)
# ===============================================
def calculer_similarite_cosinus(modele_data, mot1, mot2):
    """
    Calcule la similarité cosinus entre deux mots (COMME VOTRE CODE D'ENTRAÎNEMENT)
    """
    if modele_data is None:
        return None
    
    model = modele_data['model']
    word_to_idx = modele_data['word_to_idx']
    
    # Vérifier si les mots sont dans le vocabulaire
    if mot1 not in word_to_idx:
        st.warning(f"⚠️ Mot '{mot1}' non trouvé dans le vocabulaire")
        return None
    if mot2 not in word_to_idx:
        st.warning(f"⚠️ Mot '{mot2}' non trouvé dans le vocabulaire")
        return None
    
    idx1 = word_to_idx[mot1]
    idx2 = word_to_idx[mot2]
    
    with torch.no_grad():
        # Récupérer les embeddings (vecteurs centre)
        vec1 = model.center_embeddings(torch.LongTensor([idx1]))
        vec2 = model.center_embeddings(torch.LongTensor([idx2]))
        
        # Calculer la SIMILARITÉ COSINUS (comme dans votre code d'entraînement)
        similarite = torch.cosine_similarity(vec1, vec2).item()
    
    return similarite

def calculer_similarite_produit_scalaire(modele_data, mot1, mot2):
    """
    Calcule la similarité avec le produit scalaire (VERSION ALTERNATIVE)
    """
    if modele_data is None:
        return None
    
    model = modele_data['model']
    word_to_idx = modele_data['word_to_idx']
    
    if mot1 not in word_to_idx or mot2 not in word_to_idx:
        return None
    
    idx1 = word_to_idx[mot1]
    idx2 = word_to_idx[mot2]
    
    with torch.no_grad():
        vec1 = model.center_embeddings(torch.LongTensor([idx1]))
        vec2 = model.center_embeddings(torch.LongTensor([idx2]))
        similarite = torch.dot(vec1.squeeze(), vec2.squeeze()).item()
    
    return similarite

def trouver_mots_similaires(modele_data, mot, top_k=10, methode='cosinus'):
    """
    Trouve les mots les plus similaires à un mot donné
    
    Args:
        methode: 'cosinus' (par défaut, comme votre code) ou 'produit_scalaire'
    """
    if modele_data is None:
        return []
    
    model = modele_data['model']
    word_to_idx = modele_data['word_to_idx']
    idx_to_word = modele_data['idx_to_word']
    
    # Vérifier si le mot est dans le vocabulaire
    if mot not in word_to_idx:
        st.warning(f"⚠️ Mot '{mot}' non trouvé dans le vocabulaire")
        return []
    
    mot_idx = word_to_idx[mot]
    
    with torch.no_grad():
        # Récupérer l'embedding du mot cible
        target_vec = model.center_embeddings(torch.LongTensor([mot_idx])).squeeze(0)
        
        # Calculer les similarités avec tous les mots
        similarities = []
        for idx, other_word in idx_to_word.items():
            if idx != mot_idx:  # Exclure le mot lui-même
                other_vec = model.center_embeddings(torch.LongTensor([idx])).squeeze(0)
                
                if methode == 'cosinus':
                    # SIMILARITÉ COSINUS (comme votre code d'entraînement)
                    similarity = torch.cosine_similarity(
                        target_vec.unsqueeze(0),
                        other_vec.unsqueeze(0)
                    ).item()
                else:
                    # PRODUIT SCALAIRE (version alternative)
                    similarity = torch.dot(target_vec, other_vec).item()
                
                similarities.append((other_word, similarity))
        
        # Trier par similarité décroissante
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Retourner les top_k
        return similarities[:top_k]

# ===============================================
# INTERFACE STREAMLIT
# ===============================================
def main():
    # Initialisation des variables de session
    if 'modele_data' not in st.session_state:
        st.session_state.modele_data = None
    if 'resultats_similarite' not in st.session_state:
        st.session_state.resultats_similarite = []
    if 'similarite_mot_with' not in st.session_state:
        st.session_state.similarite_mot_with = None
    
    # Sidebar pour la configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Chemin du modèle
        chemin_modele = st.text_input(
            "Chemin du modèle SkipGram (.pth):",
            value="skipgram_model.pth",
            help="Chemin vers le fichier modèle sauvegardé depuis la partie 1"
        )
        
        # CHOIX DE LA MÉTHODE DE SIMILARITÉ
        st.markdown("---")
        st.subheader("📐 Méthode de Similarité")
        methode_similarite = st.radio(
            "Choisissez la méthode:",
            options=['cosinus', 'produit_scalaire'],
            index=0,  # Cosinus par défaut (comme votre code)
            help="Cosinus = même résultats que votre code d'entraînement. Produit scalaire = version alternative."
        )
        st.session_state.methode_similarite = methode_similarite
        
        if methode_similarite == 'cosinus':
            st.info("✅ Similarité cosinus (recommandé - comme votre code d'entraînement)")
        else:
            st.warning("⚠️ Produit scalaire (résultats différents de votre code d'entraînement)")
        
        # Information sur le modèle
        if st.session_state.modele_data:
            st.markdown("---")
            st.subheader("📊 Modèle Chargé")
            st.markdown(f"""
            - **Vocabulaire**: {st.session_state.modele_data['vocab_size']} mots
            - **Embedding Dim**: {st.session_state.modele_data['embedding_dim']}
            - **Mots connus**: {len(st.session_state.modele_data['word_to_idx'])}
            - **Méthode**: {methode_similarite.upper()}
            """)
        
        st.markdown("---")
        
        # Bouton pour charger le modèle
        if st.button("🧠 Charger le Modèle SkipGram", type="primary", use_container_width=True):
            with st.spinner("Chargement du modèle..."):
                modele_data = charger_modele_skipgram(chemin_modele)
                if modele_data:
                    st.session_state.modele_data = modele_data
                    st.balloons()
    
    # Zone principale
    if not st.session_state.modele_data:
        st.info("👈 Veuillez d'abord charger le modèle SkipGram depuis la sidebar")
        
        st.markdown("""
        ### 📋 Instructions :
        
        1. **Chargez le modèle** en spécifiant le chemin vers `skipgram_model.pth` (sidebar gauche)
        2. **Choisissez la méthode** : Cosinus (recommandé) ou Produit scalaire
        3. **Utilisez les boutons** pour explorer les similarités
        
        ### ⚠️ Important - Différence entre les méthodes :
        
        **Similarité Cosinus** (recommandée) :
        - ✅ Même résultats que votre code d'entraînement
        - ✅ Normalise les vecteurs (ignore la magnitude)
        - ✅ Valeurs entre -1 et 1
        - 📝 Formule : `cos(θ) = (A·B) / (||A|| ||B||)`
        
        **Produit Scalaire** :
        - ⚠️ Résultats différents de votre code
        - ⚠️ Sensible à la magnitude des vecteurs
        - ⚠️ Valeurs non bornées
        - 📝 Formule : `A·B`
        """)
    else:
        # ===============================================
        # LES 3 BOUTONS DEMANDÉS DANS LE TP
        # ===============================================
        st.header("🎯 Fonctionnalités - Partie 2")
        
        # Afficher la méthode active
        methode = st.session_state.methode_similarite
        if methode == 'cosinus':
            st.success(f"🎯 Méthode active : **Similarité Cosinus** (comme votre code d'entraînement)")
        else:
            st.warning(f"⚠️ Méthode active : **Produit Scalaire** (résultats différents)")
        
        # Ligne de boutons
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.button(
                "⚡ **SGNS**",
                disabled=True,
                help="Le modèle Skip-Gram avec Negative Sampling est déjà entraîné et chargé",
                use_container_width=True
            )
        
        with col2:
            if st.button("🔍 **Best similar words**", use_container_width=True):
                if 'show_similarity' in st.session_state:
                    del st.session_state.show_similarity
                st.session_state.show_similar_words = True
        
        with col3:
            if st.button("📐 **Similarity**", use_container_width=True):
                if 'show_similar_words' in st.session_state:
                    del st.session_state.show_similar_words
                st.session_state.show_similarity = True
        
        st.markdown("---")
        
        # ===============================================
        # FONCTIONNALITÉ 1: BEST SIMILAR WORDS
        # ===============================================
        if 'show_similar_words' in st.session_state and st.session_state.show_similar_words:
            st.subheader("🔍 Mots les plus similaires")
            st.markdown(f"Entrez un mot pour trouver les 10 mots les plus similaires (méthode: **{methode}**)")
            
            mot_cible = st.text_input(
                "Mot:",
                value="",
                placeholder="Exemple: particle, algorithm, data...",
                help="Entrez le mot pour lequel vous voulez trouver les mots similaires",
                key="similar_words_input"
            )
            
            if st.button("🔎 Trouver les mots similaires", type="primary", key="find_similar_btn"):
                if not mot_cible.strip():
                    st.warning("⚠️ Veuillez entrer un mot")
                else:
                    with st.spinner("Calcul des similarités..."):
                        similar_words = trouver_mots_similaires(
                            st.session_state.modele_data,
                            mot_cible.lower(),
                            top_k=10,
                            methode=methode
                        )
                        st.session_state.resultats_similarite = similar_words
            
            # Afficher les résultats
            if st.session_state.resultats_similarite:
                st.markdown(f"### 📊 Top 10 mots similaires à **'{mot_cible}'**")
                
                # Créer un DataFrame pour l'affichage
                df_similar = pd.DataFrame(
                    st.session_state.resultats_similarite,
                    columns=['Mot', f'Similarité ({methode})']
                )
                
                # Afficher avec barres de progression
                col_name = f'Similarité ({methode})'
                st.dataframe(
                    df_similar,
                    column_config={
                        col_name: st.column_config.ProgressColumn(
                            "Score de similarité",
                            help=f"Score calculé avec {methode}",
                            format="%.4f",
                            min_value=float(df_similar[col_name].min()) if methode == 'cosinus' else -1.0,
                            max_value=float(df_similar[col_name].max()) if methode == 'cosinus' else 1.0,
                        )
                    },
                    use_container_width=True,
                    hide_index=True
                )
                
                # Graphique
                st.markdown("#### 📈 Visualisation des similarités")
                df_similar[col_name] = df_similar[col_name].astype(float)
                st.bar_chart(df_similar.set_index('Mot')[col_name])
        
        # ===============================================
        # FONCTIONNALITÉ 2: SIMILARITY AVEC "WITH [MOT]"
        # ===============================================
        if 'show_similarity' in st.session_state and st.session_state.show_similarity:
            st.subheader("📐 Similarité entre un mot et 'with [mot]'")
            st.markdown(f"""
            **Explication :**  
            Entrez un mot, et le système calcule la similarité entre :
            - Le mot que vous entrez  
            - Le mot "with" suivi de votre mot (ex: "with particle")
            
            **Méthode utilisée :** {methode.upper()}
            """)
            
            mot_utilisateur = st.text_input(
                "Entrez un mot:",
                value="",
                placeholder="Exemple: particle, algorithm, data...",
                help="Mot pour calculer la similarité avec 'with [votre mot]'",
                key="similarity_input"
            )
            
            if st.button("🧮 Calculer la similarité", type="primary", key="calc_similarity_btn"):
                if not mot_utilisateur.strip():
                    st.warning("⚠️ Veuillez entrer un mot")
                else:
                    # Préparer les deux mots
                    mot1 = mot_utilisateur.lower()
                    mot2 = f"with {mot_utilisateur.lower()}"
                    
                    with st.spinner(f"Calcul de similarité entre '{mot1}' et '{mot2}'..."):
                        # Utiliser la méthode choisie
                        if methode == 'cosinus':
                            similarite = calculer_similarite_cosinus(
                                st.session_state.modele_data,
                                mot1,
                                mot2
                            )
                        else:
                            similarite = calculer_similarite_produit_scalaire(
                                st.session_state.modele_data,
                                mot1,
                                mot2
                            )
                        
                        if similarite is not None:
                            st.session_state.similarite_mot_with = {
                                'mot_original': mot1,
                                'mot_with': mot2,
                                'similarite': similarite,
                                'methode': methode
                            }
                        else:
                            st.info(f"Le mot '{mot2}' n'existe pas dans le vocabulaire. Calcul avec 'with' seul...")
                            if methode == 'cosinus':
                                similarite_with = calculer_similarite_cosinus(
                                    st.session_state.modele_data,
                                    mot1,
                                    "with"
                                )
                            else:
                                similarite_with = calculer_similarite_produit_scalaire(
                                    st.session_state.modele_data,
                                    mot1,
                                    "with"
                                )
                            
                            if similarite_with is not None:
                                st.session_state.similarite_mot_with = {
                                    'mot_original': mot1,
                                    'mot_with': "with",
                                    'similarite': similarite_with,
                                    'methode': methode,
                                    'note': f"(Note: '{mot2}' non trouvé, utilisé 'with' seul)"
                                }
            
            # Afficher le résultat
            if st.session_state.similarite_mot_with:
                sim_data = st.session_state.similarite_mot_with
                
                st.markdown("### 📊 Résultat de similarité")
                
                # Afficher la comparaison
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric(
                        "Mot original",
                        sim_data['mot_original']
                    )
                with col2:
                    st.metric(
                        "Mot comparé",
                        sim_data['mot_with']
                    )
                with col3:
                    st.metric(
                        "Similarité",
                        f"{sim_data['similarite']:.6f}",
                        delta=None
                    )
                
                # Afficher une note si nécessaire
                if 'note' in sim_data:
                    st.info(sim_data['note'])
                
                # Score de similarité avec barre de progression
                st.markdown("#### 📏 Score de similarité")
                
                # Normaliser pour l'affichage
                if methode == 'cosinus':
                    # Cosinus: de -1 à 1, normaliser à 0-1
                    normalized_score = (sim_data['similarite'] + 1) / 2
                else:
                    # Produit scalaire: normalisation approximative
                    max_possible = 10.0
                    min_possible = -10.0
                    normalized_score = max(0.0, min(1.0, 
                        (sim_data['similarite'] - min_possible) / (max_possible - min_possible)))
                
                st.progress(
                    normalized_score,
                    text=f"Score normalisé: {normalized_score:.2%}"
                )
                
                # Détails du calcul
                with st.expander("🧮 Détails du calcul"):
                    if methode == 'cosinus':
                        st.markdown(f"""
                        **Calcul (Similarité Cosinus) :**  
                        ```
                        similarité = cos(angle) entre vecteur('{sim_data['mot_original']}') et vecteur('{sim_data['mot_with']}')
                        similarité = (A·B) / (||A|| ||B||)
                        ```
                        
                        **Valeur obtenue :** `{sim_data['similarite']:.6f}`
                        
                        **Plage de valeurs :** -1.0 à 1.0
                        
                        **Interprétation :**
                        - **0.8 à 1.0** : Très forte similarité
                        - **0.5 à 0.8** : Bonne similarité
                        - **0.0 à 0.5** : Similarité modérée
                        - **< 0.0** : Dissimilarité
                        """)
                    else:
                        st.markdown(f"""
                        **Calcul (Produit Scalaire) :**  
                        ```
                        similarité = vecteur('{sim_data['mot_original']}') · vecteur('{sim_data['mot_with']}')
                        ```
                        
                        **Valeur obtenue :** `{sim_data['similarite']:.6f}`
                        
                        **Note :** Le produit scalaire n'est pas normalisé et dépend de la magnitude des vecteurs.
                        
                        **Interprétation :**
                        - **Score > 5** : Forte similarité
                        - **Score 0-5** : Similarité modérée
                        - **Score < 0** : Faible similarité
                        """)
                
                # Interprétation
                st.markdown("#### 💡 Interprétation")
                score = sim_data['similarite']
                
                if methode == 'cosinus':
                    if score > 0.8:
                        st.success("**✅ Très forte similarité** : Mots très proches sémantiquement")
                    elif score > 0.5:
                        st.success("**👍 Bonne similarité** : Relation sémantique significative")
                    elif score > 0.2:
                        st.info("**🔗 Similarité modérée** : Quelque relation existe")
                    elif score > 0:
                        st.warning("**⚠️ Faible similarité** : Peu de relations détectées")
                    else:
                        st.error("**❌ Dissimilarité** : Mots sémantiquement éloignés")
                else:
                    if score > 7.0:
                        st.success("**✅ Très forte similarité**")
                    elif score > 3.0:
                        st.success("**👍 Bonne similarité**")
                    elif score > 0:
                        st.info("**🔗 Similarité modérée**")
                    elif score > -3.0:
                        st.warning("**⚠️ Faible similarité**")
                    else:
                        st.error("**❌ Similarité négative**")
        
        # ===============================================
        # STATISTIQUES ET EXPLORATION
        # ===============================================
        st.markdown("---")
        st.header("📋 Informations du modèle")
        
        # Statistiques rapides
        col_stats1, col_stats2, col_stats3 = st.columns(3)
        with col_stats1:
            st.metric("Vocabulaire", st.session_state.modele_data['vocab_size'])
        with col_stats2:
            st.metric("Embedding Dim", st.session_state.modele_data['embedding_dim'])
        with col_stats3:
            st.metric("Méthode", methode.upper())
        
        # Recherche rapide
        with st.expander("🔍 Rechercher un mot dans le vocabulaire"):
            search_word = st.text_input("Mot à chercher:", key="vocab_search")
            if search_word:
                word_lower = search_word.lower()
                if word_lower in st.session_state.modele_data['word_to_idx']:
                    idx = st.session_state.modele_data['word_to_idx'][word_lower]
                    st.success(f"✅ Trouvé à l'index {idx}")
                    
                    # Trouver des mots similaires
                    with st.spinner("Cherchant des mots similaires..."):
                        similaires = trouver_mots_similaires(
                            st.session_state.modele_data,
                            word_lower,
                            top_k=3,
                            methode=methode
                        )
                        if similaires:
                            st.write(f"**Mots similaires (méthode: {methode}):**")
                            for mot, score in similaires:
                                st.write(f"- {mot}: {score:.4f}")
                else:
                    st.error(f"❌ Mot non trouvé")
                    
                    # Chercher des mots similaires (par forme)
                    mots_similaires_forme = [
                        w for w in st.session_state.modele_data['word_to_idx'].keys()
                        if word_lower in w or w in word_lower
                    ][:5]
                    if mots_similaires_forme:
                        st.write("**Mots similaires (par forme):**")
                        for mot in mots_similaires_forme:
                            st.write(f"- {mot}")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: gray;'>
        <p>TP N°8 - Word Embedding (Part 2) - TALN - Master 2 SII</p>
        <p>Université des Sciences et de la Technologie Houari Boumediene</p>
        <p>Faculté d'Informatique - Département d'Intelligence Artificielle</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()