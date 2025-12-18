import os
import streamlit as st
from collections import Counter
from nltk.tokenize import RegexpTokenizer
from nltk.stem import PorterStemmer, LancasterStemmer, SnowballStemmer
import itertools
import pandas as pd
import tempfile
import zipfile

# ===============================================
# Configuration et initialisation
# ===============================================
st.set_page_config(page_title="Prédicteur N-grammes", layout="wide")

# Tokenizer regex (identique à l'original)
TOKENIZER = RegexpTokenizer(r"(?:[A-Za-z]\.)+|[A-Za-z]+[\-@]\d+(?:\.\d+)?|\d+[A-Za-z]+|\d+(?:[\.\,\-]\d+)?%?|\w+(?:[\-/]\w+)*")
L1, L2 = 0.4, 0.6  # interpolation weights (identiques)

# ===============================================
# Fonctions de traitement NLP (inchangées)
# ===============================================
def tokenize_text(text, stemmer=None):
    """Tokenisation + option de normalisation (identique à l'original)."""
    tokens = [t.lower() for t in TOKENIZER.tokenize(text)]
    if stemmer:
        tokens = [stemmer.stem(t) for t in tokens]
    return tokens

def compute_ngrams(tokens, n):
    """Calcul des n-grams (identique à l'original)."""
    if len(tokens) < n:
        return {}
    counter = Counter(tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1))
    if n == 1:
        total = len(tokens)
        return {k: (v, v/total) for k, v in counter.items()}
    if n == 2:
        uni = Counter(tokens)
        return {k: (v, v/uni[k[0]]) for k, v in counter.items()}
    if n == 3:
        bi = Counter(tuple(tokens[i:i+2]) for i in range(len(tokens)-1))
        return {k: (v, v/bi[k[:2]]) for k, v in counter.items()}
    return {}

def load_corpus(text, stemmer=None):
    """Charge le corpus à partir du texte (identique à l'original)."""
    tokens = tokenize_text(text, stemmer)
    return tokens, compute_ngrams(tokens, 1), compute_ngrams(tokens, 2), compute_ngrams(tokens, 3)

# ===============================================
# Fonctions de prédiction (identiques)
# ===============================================
def case1_predict(w1, bigrams):
    """Partie I: Prédiction après un seul mot (identique)."""
    candidates = [(w2, p) for (x, w2), (_, p) in bigrams.items() if x == w1]
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[1], reverse=True)
    return candidates[:5]

def case2_predict(words, bigrams, trigrams):
    """Partie I avec interpolation (identique)."""
    w1, w2 = words[-2], words[-1]
    candidates = set()
    for (x, y), _ in bigrams.items():
        if x == w2:
            candidates.add(y)
    for (x, y, z), _ in trigrams.items():
        if (x, y) == (w1, w2):
            candidates.add(z)
    if not candidates:
        return None
    
    scored = []
    for cand in candidates:
        _, pb = bigrams.get((w2, cand), (0, 0))
        _, pt = trigrams.get((w1, w2, cand), (0, 0))
        score = L1 * pb + L2 * pt
        scored.append((cand, score))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:5]

def case3_predict(words, bigrams):
    """Partie II: Estimation de la séquence complète (identique)."""
    if len(words) > 7:
        return None, 0, "⚠ Trop de mots (max 7 pour éviter explosion combinatoire)"
    
    best_seq, best_prob = None, 0
    for perm in itertools.permutations(words):
        total_prob = 1.0
        valid = True
        for i in range(1, len(perm)):
            _, prob = bigrams.get((perm[i-1], perm[i]), (0, 0))
            if prob == 0:
                valid = False
                break
            total_prob *= prob
        if valid and total_prob > best_prob:
            best_prob = total_prob
            best_seq = perm
    
    if best_seq:
        return best_seq, best_prob, None
    else:
        return None, 0, "Impossible d'estimer la séquence (aucune bigram valide)"

# ===============================================
# Fonctions de traitement de fichiers
# ===============================================
def extract_text_from_file(file_path):
    """Extrait le texte d'un fichier .txt ou sans extension."""
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            return f.read()
    except Exception as e:
        st.error(f"Erreur lecture {file_path}: {str(e)}")
        return ""

def process_local_corpus(input_path):
    """Traite les fichiers locaux selon le type d'entrée."""
    all_texts = []
    
    if os.path.isfile(input_path):
        # Cas 1: Fichier unique
        text = extract_text_from_file(input_path)
        if text:
            all_texts.append(("fichier_unique", text))
    
    elif os.path.isdir(input_path):
        items = os.listdir(input_path)
        has_subdirs = any(os.path.isdir(os.path.join(input_path, item)) for item in items)
        
        if has_subdirs:
            # Cas 2: Structure volumique
            for subdir in items:
                subdir_path = os.path.join(input_path, subdir)
                if os.path.isdir(subdir_path):
                    for file in os.listdir(subdir_path):
                        file_path = os.path.join(subdir_path, file)
                        if os.path.isfile(file_path):
                            text = extract_text_from_file(file_path)
                            if text and len(text.strip()) > 100:
                                all_texts.append((f"{subdir}/{file}", text))
        else:
            # Cas 3: Dossier plat
            for file in os.listdir(input_path):
                file_path = os.path.join(input_path, file)
                if os.path.isfile(file_path):
                    text = extract_text_from_file(file_path)
                    if text and len(text.strip()) > 100:
                        all_texts.append((file, text))
    
    return all_texts

# ===============================================
# Interface Streamlit
# ===============================================
st.title("🧠 Prédicteur de texte N-grammes")
st.markdown("---")

# Sidebar pour la configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Sélection du mode
    mode = st.selectbox(
        "Choisissez le mode d'importation:",
        [
            "📄 Fichier unique (un seul fichier avec tous les articles)",
            "📁 Dossier d'articles (fichiers individuels)",
            "🗂️ Structure volumique (dossiers de volumes avec sous-fichiers)"
        ]
    )
    
    # Variables de session
    if 'corpus_loaded' not in st.session_state:
        st.session_state.corpus_loaded = False
    if 'corpus_text' not in st.session_state:
        st.session_state.corpus_text = ""
    if 'ngram_data' not in st.session_state:
        st.session_state.ngram_data = {}
    
    # Configuration selon le mode
    st.subheader("Paramètres")
    
    if "Fichier unique" in mode:
        input_path = st.text_input(
            "Chemin du fichier:",
            placeholder="C:/chemin/vers/fichier.txt ou fichier sans extension",
            help="Un seul fichier contenant tous les articles"
        )
    elif "Structure volumique" in mode:
        input_path = st.text_input(
            "Chemin du dossier principal:",
            placeholder="C:/chemin/vers/dossier_principal/",
            help="Dossier contenant plusieurs sous-dossiers (volumes)"
        )
    else:  # Dossier d'articles
        input_path = st.text_input(
            "Chemin du dossier d'articles:",
            placeholder="C:/chemin/vers/dossier_articles/",
            help="Dossier contenant directement les fichiers d'articles"
        )
    
    # Options de normalisation
    stemmer_choice = st.selectbox(
        "Stemmer (normalisation):",
        ["None", "Porter", "Lancaster", "Snowball"],
        help="Méthode de normalisation des mots"
    )
    
    # Bouton de chargement
    if st.button("📥 Charger le corpus", type="primary", use_container_width=True):
        if not input_path or not os.path.exists(input_path):
            st.error("❌ Chemin invalide ou non spécifié")
        else:
            with st.spinner("Chargement et traitement du corpus..."):
                try:
                    all_texts = process_local_corpus(input_path)
                    if not all_texts:
                        st.error("Aucun texte valide trouvé")
                    else:
                        # Concaténer tous les textes en un seul corpus
                        combined_text = "\n\n".join([text for _, text in all_texts])
                        st.session_state.corpus_text = combined_text
                        st.session_state.num_documents = len(all_texts)
                        st.session_state.document_names = [name for name, _ in all_texts]
                        st.session_state.corpus_loaded = True
                        st.success(f"✅ {len(all_texts)} documents chargés")
                except Exception as e:
                    st.error(f"Erreur: {str(e)}")
    
    # Informations sur les poids d'interpolation
    st.markdown("---")
    st.info(f"""
    **Paramètres d'interpolation:**
    - L1 (bigrammes) = {L1}
    - L2 (trigrammes) = {L2}
    
    Score = L1 × P(bigram) + L2 × P(trigram)
    """)

# Zone principale
if st.session_state.corpus_loaded:
    # Charger le stemmer
    stemmer = None
    if stemmer_choice == "Porter":
        stemmer = PorterStemmer()
    elif stemmer_choice == "Lancaster":
        stemmer = LancasterStemmer()
    elif stemmer_choice == "Snowball":
        stemmer = SnowballStemmer("english")
    
    # Calculer les n-grams (identique à la logique originale)
    with st.spinner("Calcul des n-grams..."):
        tokens, uni, bi, tri = load_corpus(st.session_state.corpus_text, stemmer)
        st.session_state.ngram_data = {
            'tokens': tokens,
            'unigrams': uni,
            'bigrams': bi,
            'trigrams': tri
        }
    
    # Afficher les statistiques
    st.header("📊 Statistiques du corpus")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Documents", st.session_state.num_documents)
    with col2:
        st.metric("Tokens", len(tokens))
    with col3:
        st.metric("Unigrams", len(uni))
    with col4:
        st.metric("Bigrams", len(bi))
    
    # Liste des documents chargés
    with st.expander("📋 Voir la liste des documents chargés"):
        for i, doc_name in enumerate(st.session_state.document_names, 1):
            st.write(f"{i}. {doc_name}")
    
    st.markdown("---")
    
    # Interface de prédiction
    st.header("🔮 Prédiction de texte")
    
    # Entrée utilisateur
    user_input = st.text_input(
        "Entrez votre texte:",
        placeholder="Tapez un ou plusieurs mots...",
        help="Entrez au moins un mot pour obtenir des suggestions"
    )
    
    # Bouton de prédiction
    if st.button("🎯 Analyser et prédire", type="primary"):
        if not user_input.strip():
            st.warning("⚠️ Veuillez entrer au moins un mot")
        else:
            # Tokeniser l'entrée utilisateur
            input_tokens = tokenize_text(user_input, stemmer)
            
            # Créer des onglets pour chaque partie
            tab1, tab2, tab3 = st.tabs(["📝 Partie I (un mot)", "🎯 Partie I (interpolation)", "🔢 Séquence la plus probable"])
            
            with tab1:
                st.subheader("Prédiction après un seul mot")
                if len(input_tokens) >= 1:
                    results = case1_predict(input_tokens[-1], bi)
                    if results:
                        df = pd.DataFrame(results, columns=["Mot suivant", "Probabilité"])
                        st.dataframe(df, hide_index=True, use_container_width=True)
                        
                        # Graphique
                        st.bar_chart(df.set_index("Mot suivant")["Probabilité"])
                    else:
                        st.info(f"Aucune suggestion trouvée pour '{input_tokens[-1]}'")
                else:
                    st.info("Entrez au moins un mot pour cette fonctionnalité")
            
            with tab2:
                st.subheader("Prédiction avec interpolation")
                if len(input_tokens) >= 2:
                    results = case2_predict(input_tokens, bi, tri)
                    if results:
                        df = pd.DataFrame(results, columns=["Mot suivant", "Score"])
                        df["Score"] = df["Score"].round(6)
                        st.dataframe(df, hide_index=True, use_container_width=True)
                        
                        # Graphique
                        st.bar_chart(df.set_index("Mot suivant")["Score"])
                        
                        # Détail du calcul
                        with st.expander("📐 Voir le détail des calculs"):
                            w1, w2 = input_tokens[-2], input_tokens[-1]
                            st.write(f"Derniers mots: '{w1}' '{w2}'")
                            st.write(f"**Formule:** Score = {L1} × P(bigram) + {L2} × P(trigram)")
                            for word, score in results:
                                _, pb = bi.get((w2, word), (0, 0))
                                _, pt = tri.get((w1, w2, word), (0, 0))
                                st.write(f"**{word}:** {L1}×{pb:.4f} + {L2}×{pt:.4f} = {score:.4f}")
                    else:
                        st.info(f"Aucune suggestion trouvée pour la séquence '{' '.join(input_tokens[-2:])}'")
                else:
                    st.info("Entrez au moins deux mots pour cette fonctionnalité")
            
            with tab3:
                st.subheader("Estimation de la séquence complète")
                if len(input_tokens) >= 2:
                    best_seq, best_prob, error_msg = case3_predict(input_tokens, bi)
                    
                    if error_msg:
                        st.warning(error_msg)
                    elif best_seq:
                        st.success(f"**Séquence la plus probable:**")
                        st.markdown(f"### `{' '.join(best_seq)}`")
                        st.info(f"**Probabilité:** {best_prob:.8f}")
                        
                        # Détail des probabilités
                        with st.expander("📊 Détail des calculs"):
                            st.write("**Probabilités des bigrammes:**")
                            details = []
                            total_prob = 1.0
                            for i in range(1, len(best_seq)):
                                w1, w2 = best_seq[i-1], best_seq[i]
                                _, prob = bi.get((w1, w2), (0, 0))
                                details.append({
                                    "Bigramme": f"{w1} → {w2}",
                                    "Probabilité": f"{prob:.6f}"
                                })
                                total_prob *= prob
                            
                            df_details = pd.DataFrame(details)
                            st.dataframe(df_details, hide_index=True, use_container_width=True)
                            st.write(f"**Probabilité totale:** {total_prob:.8f}")
                    else:
                        st.info("Aucune séquence valide trouvée")
                else:
                    st.info("Entrez au moins deux mots pour cette fonctionnalité")
    
    # Section pour explorer les n-grams
    st.markdown("---")
    st.header("🔍 Exploration des n-grams")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        ngram_type = st.selectbox("Type de n-gram:", ["Bigrammes", "Trigrammes"])
        search_term = st.text_input("Rechercher:", placeholder="Mot ou séquence...")
        limit_results = st.slider("Nombre de résultats:", 10, 100, 20)
    
    with col2:
        if ngram_type == "Bigrammes":
            data = bi
            display_data = [(f"{w1} {w2}", freq, prob) for (w1, w2), (freq, prob) in data.items()]
        else:
            data = tri
            display_data = [(f"{w1} {w2} {w3}", freq, prob) for (w1, w2, w3), (freq, prob) in data.items()]
        
        # Filtrer par recherche
        if search_term:
            display_data = [item for item in display_data if search_term.lower() in item[0].lower()]
        
        # Trier par fréquence
        display_data.sort(key=lambda x: x[1], reverse=True)
        display_data = display_data[:limit_results]
        
        if display_data:
            df_explore = pd.DataFrame(display_data, columns=["N-gram", "Fréquence", "Probabilité"])
            st.dataframe(df_explore, hide_index=True, use_container_width=True)
        else:
            st.info("Aucun résultat trouvé")
    
    # Export des données
    st.markdown("---")
    st.header("💾 Export des données")
    
    if st.button("📤 Exporter les n-grams"):
        with tempfile.TemporaryDirectory() as tmpdir:
            # Exporter les unigrammes
            uni_file = os.path.join(tmpdir, "unigrams.txt")
            with open(uni_file, 'w', encoding='utf-8') as f:
                for word, (freq, prob) in uni.items():
                    f.write(f"{word}\t{freq}\t{prob:.6f}\n")
            
            # Exporter les bigrammes
            bi_file = os.path.join(tmpdir, "bigrams.txt")
            with open(bi_file, 'w', encoding='utf-8') as f:
                for (w1, w2), (freq, prob) in bi.items():
                    f.write(f"{w1} {w2}\t{freq}\t{prob:.6f}\n")
            
            # Exporter les trigrammes
            tri_file = os.path.join(tmpdir, "trigrams.txt")
            with open(tri_file, 'w', encoding='utf-8') as f:
                for (w1, w2, w3), (freq, prob) in tri.items():
                    f.write(f"{w1} {w2} {w3}\t{freq}\t{prob:.6f}\n")
            
            # Créer un fichier ZIP
            zip_path = os.path.join(tmpdir, "ngrams_export.zip")
            with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for file in [uni_file, bi_file, tri_file]:
                    zipf.write(file, os.path.basename(file))
            
            # Télécharger
            with open(zip_path, 'rb') as f:
                st.download_button(
                    label="📥 Télécharger l'export (ZIP)",
                    data=f,
                    file_name="ngrams_export.zip",
                    mime="application/zip"
                )

else:
    # Page d'accueil avec instructions
    st.markdown("""
    ## 📋 Instructions d'utilisation
    
    ### **Objectif:**
    Cet outil utilise des modèles n-grammes pour prédire le mot suivant dans une séquence,
    en combinant les probabilités des bigrammes et trigrammes avec interpolation.
    
    ### **3 modes d'importation:**
    
    1. **📄 Fichier unique**
       - Un seul fichier contenant tous les articles
       - Formats: .txt ou fichier sans extension
       - Exemple: `corpus_complet.txt`
    
    2. **📁 Dossier d'articles**
       - Un dossier contenant des fichiers individuels
       - Chaque fichier = un article
       - Exemple:
       ```
       articles/
       ├── doc1.txt
       ├── doc2
       └── doc3.txt
       ```
    
    3. **🗂️ Structure volumique**
       - Dossier principal avec sous-dossiers (volumes)
       - Chaque sous-dossier contient des fichiers
       - Exemple:
       ```
       volumes/
       ├── volume_1/
       │   ├── article1.txt
       │   └── article2
       ├── volume_2/
       │   └── article3.txt
       ```
    
    ### **Algorithme (identique à l'original):**
    
    **Partie I - Prédiction simple:**
    - Après 1 mot: utilise les bigrammes `P(w2|w1)`
    
    **Partie I - Interpolation:**
    - Après 2 mots: `Score = 0.4 × P(w3|w2) + 0.6 × P(w3|w1,w2)`
    
    **Partie II - Séquence complète:**
    - Teste toutes les permutations des mots
    - Calcule la probabilité totale `Π P(wi|wi-1)`
    - Retourne la séquence avec la probabilité maximale
    
    ### **Étapes:**
    1. Sélectionnez le mode d'importation
    2. Entrez le chemin du fichier/dossier
    3. Cliquez sur "Charger le corpus"
    4. Entrez du texte et cliquez sur "Analyser et prédire"
    """)
    
    # Exemple d'utilisation
    with st.expander("🎯 Exemple d'utilisation"):
        st.markdown("""
        **Corpus:** "le chat mange la souris le chat dort"
        
        **Entrée:** "le chat"
        
        **Résultats:**
        - Partie I (interpolation): suggestions "mange", "dort"
        - Partie II: teste "chat le", "le chat", etc.
        - Séquence optimale: "le chat mange la souris"
        """)

# Note sur les dépendances
st.sidebar.markdown("---")
st.sidebar.markdown("### 📦 Dépendances nécessaires")
st.sidebar.code("""
pip install streamlit nltk pandas
""")