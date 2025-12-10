import streamlit as st
import pandas as pd
import numpy as np
from collections import Counter
import re
import io
import os
from pathlib import Path

# Configuration de la page
st.set_page_config(
    page_title="Skip-Gram Training Dataset Generator",
    page_icon="🎯",
    layout="wide"
)

# Titre principal
st.title("🎯 Skip-Gram Training Dataset Generator")
st.markdown("**TP N°8 - Word Embedding (Part 1) - TALN - Master 2 SII**")
st.markdown("---")

# Fonctions de traitement
def preprocess_text(text):
    """Prétraiter le texte"""
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    tokens = text.strip().split()
    return [token for token in tokens if token]

def load_all_articles_from_folder(folder_path):
    """Charger tous les articles du dossier All-in-many"""
    all_text = []
    articles_loaded = []
    
    if not os.path.exists(folder_path):
        return None, []
    
    # Lister tous les fichiers dans le dossier
    files = sorted([f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))])
    
    for filename in files:
        # Vérifier si c'est un article (format: Article_X_Volume_Y)
        if filename.startswith('Article_'):
            file_path = os.path.join(folder_path, filename)
            try:
                # Essayer de lire comme texte UTF-8
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                    if content.strip():  # Si le fichier n'est pas vide
                        all_text.append(content)
                        articles_loaded.append(filename)
            except:
                # Si échec, essayer avec latin-1
                try:
                    with open(file_path, 'r', encoding='latin-1', errors='ignore') as f:
                        content = f.read()
                        if content.strip():
                            all_text.append(content)
                            articles_loaded.append(filename)
                except:
                    st.warning(f"⚠️ Impossible de lire: {filename}")
                    continue
    
    if all_text:
        return '\n\n'.join(all_text), articles_loaded
    return None, []

def calculate_probabilities(word_counts, alpha=0.75):
    """Calculer les probabilités avec alpha"""
    total = sum(count**alpha for count in word_counts.values())
    probs = {}
    cumulative = []
    cum_sum = 0
    
    for word, count in word_counts.items():
        prob = (count**alpha) / total
        probs[word] = prob
        cum_sum += prob
        cumulative.append({'word': word, 'cum_prob': cum_sum})
    
    return probs, cumulative

def sample_negative_word(cumulative, exclude_words):
    """Échantillonner un mot négatif"""
    for _ in range(100):
        rand = np.random.random()
        for item in cumulative:
            if rand <= item['cum_prob']:
                if item['word'] not in exclude_words:
                    return item['word']
                break
    
    # Fallback
    available = [item['word'] for item in cumulative if item['word'] not in exclude_words]
    if available:
        return np.random.choice(available)
    return None

def generate_positive_examples(tokens, window_size):
    """Générer les exemples positifs"""
    positives = []
    
    for i, center_word in enumerate(tokens):
        start = max(0, i - window_size)
        end = min(len(tokens), i + window_size + 1)
        
        for j in range(start, end):
            if j != i:
                positives.append({
                    'center': center_word,
                    'context': tokens[j],
                    'label': 1
                })
    
    return positives

def generate_negative_examples(positives, cumulative, k):
    """Générer les exemples négatifs"""
    negatives = []
    
    for pos in positives:
        for _ in range(k):
            neg_word = sample_negative_word(cumulative, [pos['center'], pos['context']])
            if neg_word:
                negatives.append({
                    'center': pos['center'],
                    'context': neg_word,
                    'label': 0
                })
    
    return negatives

def organize_by_center(positives, negatives):
    """Organiser les exemples par mot centre"""
    by_center = {}
    
    for pos in positives:
        center = pos['center']
        if center not in by_center:
            by_center[center] = {'positives': [], 'negatives': []}
        by_center[center]['positives'].append(pos['context'])
    
    for neg in negatives:
        center = neg['center']
        if center not in by_center:
            by_center[center] = {'positives': [], 'negatives': []}
        by_center[center]['negatives'].append(neg['context'])
    
    return by_center

# Sidebar - Configuration
st.sidebar.header("⚙️ Configuration")

# Chemin du dossier All-in-many
st.sidebar.subheader("📁 Dossier des Articles")
folder_path = st.sidebar.text_input(
    "Chemin du dossier All-in-many:",
    value="./All-in-many",
    help="Entrez le chemin vers votre dossier contenant les 117 articles"
)

# Vérifier si le dossier existe
folder_exists = os.path.exists(folder_path)
if folder_exists:
    file_count = len([f for f in os.listdir(folder_path) if f.startswith('Article_')])
    st.sidebar.success(f"✅ Dossier trouvé! {file_count} articles détectés")
else:
    st.sidebar.error("❌ Dossier introuvable!")
    st.sidebar.info("💡 Exemples de chemins:\n- `./All-in-many`\n- `C:/Users/Nom/Documents/All-in-many`\n- `/home/user/All-in-many`")

st.sidebar.markdown("---")

# Paramètres
st.sidebar.subheader("🎛️ Paramètres du Modèle")
window_size = st.sidebar.slider("🪟 Taille de la fenêtre (Window Size)", 1, 10, 2)
k_negative = st.sidebar.slider("❌ K (Exemples négatifs par positif)", 1, 10, 2)
alpha = st.sidebar.slider("📊 Alpha (Puissance pour sampling)", 0.5, 1.0, 0.75, 0.05)

st.sidebar.markdown("---")

# Bouton de traitement
process_button = st.sidebar.button(
    "🚀 Charger et Générer le Dataset", 
    type="primary", 
    use_container_width=True,
    disabled=not folder_exists
)

# Traitement principal
if process_button and folder_exists:
    with st.spinner("📂 Chargement de tous les articles du dossier..."):
        # Charger tous les articles
        corpus, articles_loaded = load_all_articles_from_folder(folder_path)
        
        if corpus is None or not articles_loaded:
            st.error("❌ Aucun article n'a pu être chargé depuis le dossier!")
        else:
            st.success(f"✅ {len(articles_loaded)} articles chargés avec succès!")
            
            # Afficher la liste des articles chargés
            with st.expander("📋 Voir la liste des articles chargés"):
                cols = st.columns(3)
                for idx, article in enumerate(articles_loaded):
                    cols[idx % 3].write(f"✓ {article}")
            
            with st.spinner("🔄 Traitement du corpus en cours..."):
                # Prétraitement
                tokens = preprocess_text(corpus)
                
                if not tokens:
                    st.error("❌ Le corpus est vide après prétraitement!")
                else:
                    # Compter les mots
                    word_counts = Counter(tokens)
                    vocabulary = list(word_counts.keys())
                    
                    # Calculer les probabilités
                    probs, cumulative = calculate_probabilities(word_counts, alpha)
                    
                    # Générer les exemples
                    with st.spinner("✅ Génération des exemples positifs..."):
                        positives = generate_positive_examples(tokens, window_size)
                    
                    with st.spinner("❌ Génération des exemples négatifs..."):
                        negatives = generate_negative_examples(positives, cumulative, k_negative)
                    
                    # Organiser par centre
                    by_center = organize_by_center(positives, negatives)
                    
                    # Combiner tous les exemples
                    all_examples = positives + negatives
                    
                    # Sauvegarder dans session state
                    st.session_state['results'] = {
                        'tokens': tokens,
                        'vocabulary': vocabulary,
                        'word_counts': word_counts,
                        'probabilities': probs,
                        'positives': positives,
                        'negatives': negatives,
                        'all_examples': all_examples,
                        'by_center': by_center,
                        'articles_loaded': articles_loaded,
                        'corpus_preview': corpus[:500] + "..." if len(corpus) > 500 else corpus
                    }
                    
                    st.success("✅ Dataset généré avec succès!")
                    st.balloons()

# Affichage des résultats
if 'results' in st.session_state:
    results = st.session_state['results']
    
    # Informations sur les articles chargés
    st.header("📚 Informations sur le Corpus")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("📄 Articles Chargés", len(results['articles_loaded']))
    with col2:
        st.metric("📏 Taille du Corpus", f"{len(results['corpus_preview'])} caractères (aperçu)")
    with col3:
        st.metric("🔤 Total Tokens", len(results['tokens']))
    
    # Aperçu du corpus
    with st.expander("👀 Aperçu du corpus combiné (500 premiers caractères)"):
        st.text(results['corpus_preview'])
    
    st.markdown("---")
    
    # Statistiques
    st.header("📊 Statistiques du Dataset")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("🔤 Tokens", len(results['tokens']))
    with col2:
        st.metric("📚 Vocabulaire", len(results['vocabulary']))
    with col3:
        st.metric("✅ Exemples Positifs", len(results['positives']))
    with col4:
        st.metric("❌ Exemples Négatifs", len(results['negatives']))
    
    st.markdown("---")
    
    # Vocabulaire et probabilités (Top 50 pour ne pas surcharger)
    st.header("📚 Vocabulaire & Probabilités (Top 50 mots les plus fréquents)")
    
    # Trier par fréquence
    sorted_vocab = sorted(results['vocabulary'], key=lambda w: results['word_counts'][w], reverse=True)[:50]
    
    vocab_data = []
    total_tokens = len(results['tokens'])
    for word in sorted_vocab:
        count = results['word_counts'][word]
        std_prob = count / total_tokens
        adj_prob = results['probabilities'][word]
        vocab_data.append({
            'Mot': word,
            'Fréquence': count,
            f'Probabilité Standard': f"{std_prob:.4f}",
            f'Probabilité Ajustée (α={alpha})': f"{adj_prob:.4f}"
        })
    
    vocab_df = pd.DataFrame(vocab_data)
    st.dataframe(vocab_df, use_container_width=True, hide_index=True)
    
    # Option pour voir tout le vocabulaire
    if len(results['vocabulary']) > 50:
        with st.expander(f"📖 Voir tout le vocabulaire ({len(results['vocabulary'])} mots)"):
            all_vocab_data = []
            for word in results['vocabulary']:
                count = results['word_counts'][word]
                std_prob = count / total_tokens
                adj_prob = results['probabilities'][word]
                all_vocab_data.append({
                    'Mot': word,
                    'Fréquence': count,
                    f'Probabilité Standard': f"{std_prob:.4f}",
                    f'Probabilité Ajustée (α={alpha})': f"{adj_prob:.4f}"
                })
            all_vocab_df = pd.DataFrame(all_vocab_data)
            st.dataframe(all_vocab_df, use_container_width=True, hide_index=True, height=400)
    
    st.markdown("---")
    
    # Contextes par mot centre
    st.header("🎯 Contextes par Mot Centre")
    
    selected_word = st.selectbox(
        "Sélectionnez un mot centre:",
        options=["-- Choisir un mot --"] + sorted(results['vocabulary'])
    )
    
    if selected_word != "-- Choisir un mot --" and selected_word in results['by_center']:
        data = results['by_center'][selected_word]
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("✅ Contextes POSITIFS")
            unique_positives = list(set(data['positives']))
            st.info(f"**Nombre total:** {len(data['positives'])} (dont {len(unique_positives)} uniques)")
            if unique_positives:
                for ctx in unique_positives[:20]:  # Limiter à 20 pour l'affichage
                    st.success(f"**{ctx}**")
                if len(unique_positives) > 20:
                    st.write(f"... et {len(unique_positives) - 20} autres")
            else:
                st.info("Aucun contexte positif")
        
        with col2:
            st.subheader("❌ Contextes NÉGATIFS (échantillonnés)")
            unique_negatives = list(set(data['negatives']))
            st.info(f"**Nombre total:** {len(data['negatives'])} (dont {len(unique_negatives)} uniques)")
            if unique_negatives:
                for ctx in unique_negatives[:20]:  # Limiter à 20 pour l'affichage
                    st.error(f"**{ctx}**")
                if len(unique_negatives) > 20:
                    st.write(f"... et {len(unique_negatives) - 20} autres")
            else:
                st.info("Aucun contexte négatif")
    
    st.markdown("---")
    
    # Dataset complet (avec pagination)
    st.header("📋 Dataset d'Entraînement Complet")
    
    # Options d'affichage
    col1, col2 = st.columns(2)
    with col1:
        display_limit = st.number_input(
            "Nombre d'exemples à afficher:",
            min_value=10,
            max_value=len(results['all_examples']),
            value=min(100, len(results['all_examples'])),
            step=10
        )
    with col2:
        filter_type = st.selectbox(
            "Filtrer par type:",
            ["Tous", "Positifs uniquement", "Négatifs uniquement"]
        )
    
    # Filtrer les exemples
    if filter_type == "Positifs uniquement":
        filtered_examples = [ex for ex in results['all_examples'] if ex['label'] == 1]
    elif filter_type == "Négatifs uniquement":
        filtered_examples = [ex for ex in results['all_examples'] if ex['label'] == 0]
    else:
        filtered_examples = results['all_examples']
    
    # Créer un DataFrame
    dataset_data = []
    for i, ex in enumerate(filtered_examples[:display_limit], 1):
        dataset_data.append({
            '#': i,
            'Mot Centre': ex['center'],
            'Mot Contexte': ex['context'],
            'Label': ex['label'],
            'Type': '✅ Positif' if ex['label'] == 1 else '❌ Négatif'
        })
    
    dataset_df = pd.DataFrame(dataset_data)
    
    # Colorier selon le type
    def highlight_rows(row):
        if row['Label'] == 1:
            return ['background-color: #d4edda'] * len(row)
        else:
            return ['background-color: #f8d7da'] * len(row)
    
    styled_df = dataset_df.style.apply(highlight_rows, axis=1)
    st.dataframe(styled_df, use_container_width=True, hide_index=True, height=400)
    
    if len(filtered_examples) > display_limit:
        st.info(f"📊 Affichage de {display_limit} sur {len(filtered_examples)} exemples au total")
    
    st.markdown("---")
    
    # Bouton de téléchargement
    st.header("💾 Télécharger les Résultats")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Préparer le contenu du fichier texte
        output = io.StringIO()
        output.write("=== SKIP-GRAM TRAINING DATASET ===\n\n")
        output.write(f"Articles chargés: {len(results['articles_loaded'])}\n")
        output.write(f"Total tokens: {len(results['tokens'])}\n")
        output.write(f"Vocabulaire: {len(results['vocabulary'])} mots\n")
        output.write(f"Window Size: {window_size}\n")
        output.write(f"K (Negative samples): {k_negative}\n")
        output.write(f"Alpha: {alpha}\n\n")
        
        output.write("=== ARTICLES CHARGÉS ===\n")
        for article in results['articles_loaded']:
            output.write(f"- {article}\n")
        
        output.write("\n=== TOP 100 VOCABULARY ===\n")
        top_vocab = sorted(results['vocabulary'], key=lambda w: results['word_counts'][w], reverse=True)[:100]
        for word in top_vocab:
            count = results['word_counts'][word]
            prob = results['probabilities'][word]
            output.write(f"{word}: count={count}, prob={prob:.4f}\n")
        
        output.write("\n=== POSITIVE EXAMPLES (premiers 100) ===\n")
        for i, ex in enumerate(results['positives'][:100], 1):
            output.write(f"{i}. ({ex['center']}, {ex['context']}) : {ex['label']}\n")
        
        output.write("\n=== NEGATIVE EXAMPLES (premiers 100) ===\n")
        for i, ex in enumerate(results['negatives'][:100], 1):
            output.write(f"{i}. ({ex['center']}, {ex['context']}) : {ex['label']}\n")
        
        st.download_button(
            label="📥 Télécharger le résumé (TXT)",
            data=output.getvalue(),
            file_name="skipgram_dataset_summary.txt",
            mime="text/plain",
            use_container_width=True
        )
    
    with col2:
        # Préparer le CSV complet
        csv_data = []
        for ex in results['all_examples']:
            csv_data.append({
                'center_word': ex['center'],
                'context_word': ex['context'],
                'label': ex['label']
            })
        csv_df = pd.DataFrame(csv_data)
        
        st.download_button(
            label="📥 Télécharger le dataset complet (CSV)",
            data=csv_df.to_csv(index=False),
            file_name="skipgram_complete_dataset.csv",
            mime="text/csv",
            use_container_width=True
        )

else:
    # Message d'accueil
    st.info("👈 Configurez le chemin du dossier dans la barre latérale et cliquez sur '🚀 Charger et Générer le Dataset'")
    
    st.markdown("""
    ### 📖 Instructions d'utilisation:
    
    1. **📁 Spécifiez le chemin** vers votre dossier `All-in-many` dans la barre latérale
       - Exemple: `./All-in-many` (si dans le même dossier)
       - Exemple: `C:/Users/VotreNom/Documents/All-in-many` (Windows)
       - Exemple: `/home/user/All-in-many` (Linux)
    
    2. **🎛️ Ajustez les paramètres** si nécessaire:
       - Window Size (taille de la fenêtre)
       - K (nombre d'exemples négatifs)
       - Alpha (puissance pour l'échantillonnage)
    
    3. **🚀 Cliquez sur "Charger et Générer le Dataset"**
       - L'application chargera automatiquement TOUS les fichiers du dossier
       - Même sans extension `.txt` !
       - Format détecté: `Article_X_Volume_Y`
    
    4. **📊 Explorez les résultats**:
       - Statistiques globales
       - Vocabulaire et probabilités
       - Contextes par mot centre
       - Dataset complet
    
    5. **💾 Téléchargez les résultats** en TXT ou CSV
    
    ### ✨ Fonctionnalités:
    - ✅ Charge automatiquement tous les articles du dossier
    - ✅ Supporte les fichiers SANS extension
    - ✅ Affichage identique au TP du professeur
    - ✅ Export en TXT et CSV
    - ✅ Gestion de gros corpus (117 fichiers)
    """)
    
    st.markdown("---")
    
    # Exemple visuel
    st.subheader("📂 Structure attendue du dossier:")
    st.code("""
All-in-many/
├── Article_1_Volume_18
├── Article_2_Volume_18
├── Article_3_Volume_18
├── ...
└── Article_117_Volume_XX
    """, language="text")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    <p>TP N°8 - Word Embedding - TALN - Master 2 SII</p>
    <p>Université des Sciences et de la Technologie Houari Boumediene</p>
    <p>Faculté d'Informatique - Département d'Intelligence Artificielle</p>
</div>
""", unsafe_allow_html=True)