import os
import re
import math
import streamlit as st
from collections import Counter, defaultdict
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.stem import PorterStemmer, LancasterStemmer, SnowballStemmer
import pandas as pd

# ===============================================
# Configuration et initialisation
# ===============================================
st.set_page_config(page_title="Naive Bayes Classifier", layout="wide")

# Stopwords (fallback empty set)
@st.cache_resource
def get_stopwords():
    try:
        from nltk.corpus import stopwords
        return set(stopwords.words('english'))
    except Exception:
        try:
            nltk.download('stopwords', quiet=True)
            from nltk.corpus import stopwords
            return set(stopwords.words('english'))
        except Exception:
            return set()

STOPWORDS = get_stopwords()

LABEL_MAP = {
    "1": "Metaheuristics",
    "2": "Machine & Deep Learning", 
    "3": "Combination of Metaheuristics & Machine/Deep Learning",
    "4": "Others"
}

MAX_WORDS_DISPLAY = 200

# ===============================================
# Fonctions de traitement (inchangées)
# ===============================================
def preprocess_text(text, stemmer="none"):
    if not isinstance(text, str): return ""
    text = text.replace("<s>", "").replace("</s>", "")
    tok = RegexpTokenizer(r'(?:[A-Za-z]\.)+|[A-Za-z]+[\-@]\d+(?:\.\d+)?|\d+[A-Za-z]+|\d+(?:[\.\,\-]\d+)?%?|\w+(?:[\-/]\w+)*|[\.\!\?]+')
    tokens = [t.lower() for t in tok.tokenize(text) if t.lower() not in STOPWORDS]
    
    if stemmer == "porter":
        s = PorterStemmer()
        tokens = [s.stem(t) for t in tokens]
    elif stemmer == "lancaster":
        s = LancasterStemmer()
        tokens = [s.stem(t) for t in tokens]
    elif stemmer == "snowball":
        s = SnowballStemmer("english")
        tokens = [s.stem(t) for t in tokens]
    
    return " ".join(tokens)

def _files_map(folder):
    """Map des fichiers dans un dossier"""
    m = {}
    try:
        for fname in os.listdir(folder):
            if fname.startswith("."): continue
            full = os.path.join(folder, fname)
            if os.path.isfile(full):
                base = os.path.splitext(fname)[0]
                if base not in m: m[base] = fname
    except Exception as e:
        st.error(f"Erreur listage {folder}: {e}")
    return m

def _load_single_volume_with_separate_dirs(articles_dir, labels_dir, volume_name="Volume Unique"):
    """Charge les articles et labels depuis deux dossiers séparés (mode volume unique original)"""
    am = _files_map(articles_dir)
    lm = _files_map(labels_dir)
    common = sorted(set(am).intersection(lm))
    
    articles = []
    skipped = 0
    
    for base in common:
        a_path = os.path.join(articles_dir, am[base])
        l_path = os.path.join(labels_dir, lm[base])
        
        try:
            with open(a_path, "r", encoding="utf-8") as f:
                text = f.read()
        except:
            try:
                with open(a_path, "r", encoding="latin-1") as f:
                    text = f.read()
            except:
                skipped += 1
                continue
        
        try:
            with open(l_path, "r", encoding="utf-8") as f:
                raw_label = f.read()
        except:
            try:
                with open(l_path, "r", encoding="latin-1") as f:
                    raw_label = f.read()
            except:
                skipped += 1
                continue
        
        m = re.search(r"[1-4]", raw_label)
        label = m.group(0) if m else "?"
        proc = preprocess_text(text, st.session_state.stemmer_choice)
        
        # Extraire le numéro d'article
        mnum = re.search(r"\b(\d+)\b", base)
        artnum = mnum.group(1) if mnum else "?"
        
        articles.append({
            "Article": artnum,
            "LabelNum": label,
            "LabelName": LABEL_MAP.get(label, "Unknown"),
            "Base": base,
            "Text": proc,
            "Set": "Training",
            "Volume": volume_name
        })
    
    return articles, skipped

def _load_single_volume_same_dir(volume_dir, volume_name=None):
    """Charge les articles et labels depuis un même dossier (mode multi-volumes)"""
    articles = []
    skipped = 0
    
    try:
        # Lister tous les fichiers dans le dossier
        files = os.listdir(volume_dir)
    except Exception as e:
        st.error(f"Erreur d'accès au dossier {volume_dir}: {e}")
        return articles, skipped
    
    # Séparer articles et labels
    article_files = {}
    label_files = {}
    
    for fname in files:
        if fname.startswith("."): continue
        
        base_name = os.path.splitext(fname)[0]
        
        # Si c'est un fichier de label
        if "_label" in base_name:
            # Extraire le nom de base de l'article correspondant
            article_base = base_name.replace("_label", "")
            label_files[article_base] = fname
        # Si c'est un fichier d'article (sans _label)
        elif not base_name.endswith("_label"):
            article_files[base_name] = fname
    
    # Trouver les paires correspondantes
    common_bases = set(article_files.keys()).intersection(set(label_files.keys()))
    
    for base in common_bases:
        a_path = os.path.join(volume_dir, article_files[base])
        l_path = os.path.join(volume_dir, label_files[base])
        
        try:
            with open(a_path, "r", encoding="utf-8") as f:
                text = f.read()
        except:
            try:
                with open(a_path, "r", encoding="latin-1") as f:
                    text = f.read()
            except:
                skipped += 1
                continue
        
        try:
            with open(l_path, "r", encoding="utf-8") as f:
                raw_label = f.read()
        except:
            try:
                with open(l_path, "r", encoding="latin-1") as f:
                    raw_label = f.read()
            except:
                skipped += 1
                continue
        
        m = re.search(r"[1-4]", raw_label)
        label = m.group(0) if m else "?"
        proc = preprocess_text(text, st.session_state.stemmer_choice)
        
        # Extraire le numéro d'article
        mnum = re.search(r"Article_(\d+)", base)
        artnum = mnum.group(1) if mnum else "?"
        
        # Utiliser le nom du volume fourni ou le nom du dossier
        vol_name = volume_name if volume_name else os.path.basename(volume_dir)
        
        articles.append({
            "Article": artnum,
            "LabelNum": label,
            "LabelName": LABEL_MAP.get(label, "Unknown"),
            "Base": base,
            "Text": proc,
            "Set": "Training",
            "Volume": vol_name,
            "ArticleFile": article_files[base],
            "LabelFile": label_files[base]
        })
    
    return articles, skipped

def _load_test_files(test_dir):
    """Charge les fichiers de test (identique à l'original)"""
    test_data = []
    skipped = 0
    
    try:
        files = os.listdir(test_dir)
    except Exception as e:
        return [], f"Impossible d'accéder au dossier de test: {e}"
    
    # Séparer articles et labels
    article_files = {}
    label_files = {}
    
    for fname in files:
        if fname.startswith("."): continue
        
        base_name = os.path.splitext(fname)[0]
        
        # Si c'est un fichier de label
        if "_label" in base_name:
            # Extraire le nom de base de l'article correspondant
            article_base = base_name.replace("_label", "")
            label_files[article_base] = fname
        # Si c'est un fichier d'article (sans _label)
        elif not base_name.endswith("_label"):
            article_files[base_name] = fname
    
    # Trouver les paires correspondantes
    common_bases = set(article_files.keys()).intersection(set(label_files.keys()))
    
    if not common_bases:
        return [], "Aucune paire article/label trouvée dans le dossier de test."
    
    for base in common_bases:
        article_path = os.path.join(test_dir, article_files[base])
        label_path = os.path.join(test_dir, label_files[base])
        
        # Charger l'article
        try:
            with open(article_path, "r", encoding="utf-8") as f:
                text = f.read()
        except:
            try:
                with open(article_path, "r", encoding="latin-1") as f:
                    text = f.read()
            except:
                skipped += 1
                continue
        
        # Charger le label
        try:
            with open(label_path, "r", encoding="utf-8") as f:
                raw_label = f.read()
        except:
            try:
                with open(label_path, "r", encoding="latin-1") as f:
                    raw_label = f.read()
            except:
                skipped += 1
                continue
        
        # Extraire le label numérique
        m = re.search(r"[1-4]", raw_label)
        label = m.group(0) if m else "?"
        
        # Prétraiter le texte
        proc = preprocess_text(text, st.session_state.stemmer_choice)
        
        # Extraire le numéro d'article
        mnum = re.search(r"Article_(\d+)", base)
        artnum = mnum.group(1) if mnum else "?"
        
        test_data.append({
            "Article": artnum,
            "LabelNum": label,
            "LabelName": LABEL_MAP.get(label, "Unknown"),
            "Base": base,
            "Text": proc,
            "Set": "Testing",
            "ArticleFile": article_files[base],
            "LabelFile": label_files[base]
        })
    
    return test_data, f"{skipped} fichiers ignorés"

def predict_class(text, priors, cond_probs, sum_words, vocab_size):
    """Prédit la classe d'un texte (identique à l'original)"""
    if priors is None or cond_probs is None:
        return None
    
    tokens = text.split()
    best, best_score = None, -1e99
    
    for c in priors:
        score = math.log10(priors[c][1])
        denom = sum_words[c] + vocab_size
        
        for w in tokens:
            prob = cond_probs.get(w, {}).get(c, 1/denom)
            score += math.log10(prob)
        
        if score > best_score:
            best_score = score
            best = c
    
    return best

# ===============================================
# Interface Streamlit
# ===============================================
st.title("🧠 Classificateur Naive Bayes")
st.markdown("---")

# Initialisation des variables de session
if 'training_data' not in st.session_state:
    st.session_state.training_data = []
if 'test_data' not in st.session_state:
    st.session_state.test_data = []
if 'priors' not in st.session_state:
    st.session_state.priors = None
if 'cond_probs' not in st.session_state:
    st.session_state.cond_probs = None
if 'sum_words' not in st.session_state:
    st.session_state.sum_words = None
if 'vocab_size' not in st.session_state:
    st.session_state.vocab_size = 0
if 'stemmer_choice' not in st.session_state:
    st.session_state.stemmer_choice = "none"
if 'class_counts' not in st.session_state:
    st.session_state.class_counts = None
if 'training_mode' not in st.session_state:
    st.session_state.training_mode = "single_volume"

# Sidebar pour la configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Options de normalisation
    st.subheader("Normalisation")
    stemmer_choice = st.selectbox(
        "Stemmer:",
        ["none", "porter", "lancaster", "snowball"]
    )
    st.session_state.stemmer_choice = stemmer_choice
    
    # Sélection du mode d'entraînement
    st.subheader("Mode d'entraînement")
    training_mode = st.selectbox(
        "Choisissez le mode d'entraînement:",
        ["📁 Volume unique", "📚 Multiples volumes"]
    )
    st.session_state.training_mode = training_mode
    
    # Mode d'importation
    st.subheader("Mode d'importation")
    mode = st.selectbox(
        "Choisissez le mode:",
        ["📚 Entraînement + Test", "📁 Test uniquement (modèle existant)"]
    )
    
    # Configuration selon le mode d'entraînement
    if mode == "📚 Entraînement + Test":
        if training_mode == "📁 Volume unique":
            # Mode original: un seul volume avec dossiers séparés
            st.subheader("Dossiers d'entraînement (volume unique)")
            train_articles_dir = st.text_input(
                "Dossier des articles d'entraînement:",
                placeholder="C:/chemin/vers/articles/",
                help="Dossier contenant les fichiers Article_XX.txt"
            )
            
            train_labels_dir = st.text_input(
                "Dossier des labels d'entraînement:",
                placeholder="C:/chemin/vers/labels/",
                help="Dossier contenant les fichiers Article_XX_label.txt"
            )
        
        elif training_mode == "📚 Multiples volumes":
            # Nouveau mode: multiples volumes avec articles et labels dans le même dossier
            st.subheader("Dossier des volumes d'entraînement")
            volumes_dir = st.text_input(
                "Dossier racine des volumes:",
                placeholder="C:/chemin/vers/volumes/",
                help="Dossier contenant plusieurs sous-dossiers (volumes), chacun avec ses articles et labels"
            )
    
    # Dossier de test (identique pour les deux modes)
    st.subheader("Dossier de test")
    test_dir = st.text_input(
        "Dossier de test:",
        placeholder="C:/chemin/vers/test/",
        help="Dossier contenant les fichiers de test (Article_XX.txt et Article_XX_label.txt)"
    )
    
    # Boutons d'action
    st.markdown("---")
    
    # Bouton pour charger l'entraînement
    if mode == "📚 Entraînement + Test":
        if training_mode == "📁 Volume unique":
            if st.button("📥 Charger l'entraînement", use_container_width=True):
                if not os.path.exists(train_articles_dir) or not os.path.exists(train_labels_dir):
                    st.error("Vérifiez les dossiers articles/labels d'entraînement.")
                else:
                    with st.spinner("Chargement des données d'entraînement..."):
                        articles, skipped = _load_single_volume_with_separate_dirs(
                            train_articles_dir, 
                            train_labels_dir, 
                            "Volume Unique"
                        )
                        
                        if not articles:
                            st.warning("Aucun article trouvé.")
                        else:
                            st.session_state.training_data = articles
                            st.success(f"✅ {len(articles)} articles d'entraînement chargés. Ignorés: {skipped}")
        
        elif training_mode == "📚 Multiples volumes":
            if st.button("📥 Charger l'entraînement (multi-volumes)", use_container_width=True):
                if not os.path.exists(volumes_dir):
                    st.error("Dossier racine des volumes non trouvé.")
                else:
                    with st.spinner("Recherche des volumes d'entraînement..."):
                        all_articles = []
                        total_skipped = 0
                        volumes_loaded = 0
                        
                        # Parcourir tous les sous-dossiers (volumes)
                        try:
                            volume_folders = [d for d in os.listdir(volumes_dir) 
                                           if os.path.isdir(os.path.join(volumes_dir, d))]
                        except Exception as e:
                            st.error(f"Erreur d'accès au dossier: {e}")
                            volume_folders = []
                        
                        if not volume_folders:
                            st.warning("Aucun sous-dossier (volume) trouvé.")
                        else:
                            for volume_name in volume_folders:
                                volume_path = os.path.join(volumes_dir, volume_name)
                                
                                # Charger ce volume
                                try:
                                    articles_in_volume, skipped_in_volume = _load_single_volume_same_dir(
                                        volume_path, 
                                        volume_name
                                    )
                                    
                                    if articles_in_volume:
                                        all_articles.extend(articles_in_volume)
                                        total_skipped += skipped_in_volume
                                        volumes_loaded += 1
                                        st.info(f"Volume '{volume_name}': {len(articles_in_volume)} articles")
                                        
                                except Exception as e:
                                    st.warning(f"Erreur dans le volume {volume_name}: {e}")
                                    continue
                            
                            if all_articles:
                                st.session_state.training_data = all_articles
                                st.success(f"✅ {len(all_articles)} articles chargés depuis {volumes_loaded} volumes. Ignorés: {total_skipped}")
                            else:
                                st.warning("Aucun article trouvé dans les volumes.")
    
    # Bouton pour entraîner le modèle
    if mode == "📚 Entraînement + Test":
        if st.button("🎓 Entraîner le modèle", use_container_width=True, type="primary"):
            if not st.session_state.training_data:
                st.warning("Chargez d'abord les articles d'entraînement.")
            else:
                with st.spinner("Entraînement du modèle Naive Bayes..."):
                    N = len(st.session_state.training_data)
                    class_counts = Counter()
                    class_word_counts = defaultdict(Counter)
                    vocab = set()
                    
                    for item in st.session_state.training_data:
                        c = item["LabelNum"]
                        class_counts[c] += 1
                        toks = item["Text"].split()
                        vocab.update(toks)
                        class_word_counts[c].update(toks)
                    
                    V = len(vocab)
                    st.session_state.vocab_size = V
                    st.session_state.class_counts = class_counts
                    st.session_state.sum_words = {c: sum(class_word_counts[c].values()) for c in class_word_counts}
                    
                    # Calcul des priors
                    priors = {}
                    for c in class_counts:
                        priors[c] = (class_counts[c], class_counts[c] / N)
                    st.session_state.priors = priors
                    
                    # Calcul des probabilités conditionnelles
                    cond_probs = defaultdict(dict)
                    for c in class_word_counts:
                        denom = st.session_state.sum_words[c] + V
                        for w in vocab:
                            cond_probs[w][c] = (class_word_counts[c][w] + 1) / denom
                    st.session_state.cond_probs = cond_probs
                    
                    # Afficher les statistiques
                    volumes_summary = {}
                    for article in st.session_state.training_data:
                        vol = article.get("Volume", "Single Volume")
                        volumes_summary[vol] = volumes_summary.get(vol, 0) + 1
                    
                    st.success(f"✅ Modèle entraîné avec {N} articles")
                    st.success(f"   - Vocabulaire: {V} mots")
                    st.success(f"   - Classes: {len(class_counts)}")
                    
                    if len(volumes_summary) > 1:
                        st.success(f"   - Volumes utilisés: {len(volumes_summary)}")
                        for vol_name, count in sorted(volumes_summary.items()):
                            st.success(f"     - {vol_name}: {count} articles")
    
    # Bouton pour charger et tester
    if st.button("🧪 Charger & Tester", use_container_width=True, type="secondary"):
        if st.session_state.priors is None or st.session_state.cond_probs is None:
            st.warning("Entraînez d'abord le modèle.")
        elif not test_dir or not os.path.exists(test_dir):
            st.error("Dossier de test non trouvé.")
        else:
            with st.spinner("Chargement des données de test..."):
                test_data, skipped_msg = _load_test_files(test_dir)
                
                if not test_data:
                    st.warning(f"Aucune donnée de test valide. {skipped_msg}")
                else:
                    st.session_state.test_data = test_data
                    
                    # Tester le modèle
                    results = []
                    correct = 0
                    
                    for item in test_data:
                        pred = predict_class(
                            item["Text"],
                            st.session_state.priors,
                            st.session_state.cond_probs,
                            st.session_state.sum_words,
                            st.session_state.vocab_size
                        )
                        
                        ok = (pred == item["LabelNum"])
                        if ok:
                            correct += 1
                        
                        results.append({
                            "Article": item["Article"],
                            "Vrai Label": item["LabelName"],
                            "Label Prédit": LABEL_MAP.get(pred, "Unknown"),
                            "Correct": "✓" if ok else "✗",
                            "Base": item["Base"],
                            "Vrai Num": item["LabelNum"],
                            "Prédit Num": pred
                        })
                    
                    st.session_state.test_results = results
                    st.session_state.accuracy = correct / len(test_data) if test_data else 0
                    st.success(f"✅ {len(test_data)} articles testés. Précision: {st.session_state.accuracy:.2%}")

# Zone principale
if mode == "📚 Entraînement + Test" and st.session_state.training_data:
    st.header("📊 Données d'entraînement")
    
    # Afficher le mode d'entraînement
    mode_text = "Volume unique" if training_mode == "📁 Volume unique" else "Multiples volumes"
    st.info(f"Mode d'entraînement: {mode_text}")
    
    # Statistiques des données d'entraînement
    if training_mode == "📚 Multiples volumes":
        # Afficher un résumé par volume
        volumes_summary = {}
        for article in st.session_state.training_data:
            vol = article.get("Volume", "Inconnu")
            volumes_summary[vol] = volumes_summary.get(vol, 0) + 1
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Articles totaux", len(st.session_state.training_data))
        with col2:
            st.metric("Volumes", len(volumes_summary))
        with col3:
            avg_per_volume = len(st.session_state.training_data) / len(volumes_summary) if volumes_summary else 0
            st.metric("Moyenne/volume", f"{avg_per_volume:.1f}")
        
        # Afficher la répartition par volume
        with st.expander("📈 Répartition par volume"):
            volumes_df = pd.DataFrame([
                {"Volume": vol, "Articles": count, "Pourcentage": f"{(count/len(st.session_state.training_data))*100:.1f}%"}
                for vol, count in volumes_summary.items()
            ])
            st.dataframe(volumes_df.sort_values("Articles", ascending=False), use_container_width=True)
    
    # Recherche dans le tableau d'entraînement
    col1, col2 = st.columns([3, 1])
    with col1:
        search_train = st.text_input("Rechercher dans l'entraînement:", placeholder="Article, label, base ou volume...")
    with col2:
        sort_train_by = st.selectbox("Trier par:", ["Article", "LabelNum", "Volume", "Base"])
    
    # Filtrer et trier les données
    train_df = pd.DataFrame(st.session_state.training_data)
    if not train_df.empty:
        if search_train:
            mask = train_df.apply(lambda row: row.astype(str).str.contains(search_train, case=False).any(), axis=1)
            train_df = train_df[mask]
        
        if sort_train_by == "Article":
            train_df["Article_Num"] = pd.to_numeric(train_df["Article"], errors='coerce')
            train_df = train_df.sort_values("Article_Num").drop(columns=["Article_Num"])
        else:
            train_df = train_df.sort_values(sort_train_by)
        
        # Colonnes à afficher - AJOUT DE LA COLONNE VOLUME
        display_columns = ["Article", "LabelNum", "LabelName", "Base", "Set"]
        if "Volume" in train_df.columns:
            display_columns.insert(3, "Volume")  # Insérer Volume après LabelName
        
        # Afficher le tableau avec la colonne Volume
        st.dataframe(
            train_df[display_columns],
            use_container_width=True,
            height=300
        )
    
    # Afficher les résultats de l'entraînement si disponibles
    if st.session_state.priors is not None:
        st.markdown("---")
        st.header("📈 Résultats de l'entraînement")
        
        # Priors
        st.subheader("Probabilités a priori")
        priors_data = []
        for c in sorted(st.session_state.priors.keys(), key=lambda x: int(x) if x.isdigit() else x):
            priors_data.append({
                "Classe": c,
                "Label": LABEL_MAP.get(c, "Unknown"),
                "Count": st.session_state.priors[c][0],
                "Prior": f"{st.session_state.priors[c][1]:.6f}"
            })
        
        priors_df = pd.DataFrame(priors_data)
        st.dataframe(priors_df, use_container_width=True)
        
        # Probabilités conditionnelles (TOUS les mots)
        if st.session_state.cond_probs is not None and st.session_state.class_counts is not None:
            # Récupérer TOUS les mots du vocabulaire
            all_words = list(st.session_state.cond_probs.keys())
            st.subheader(f"Probabilités conditionnelles ({len(all_words)} mots)")
            
            # Créer le DataFrame avec TOUS les mots
            cond_data = []
            classes = sorted(st.session_state.class_counts.keys(), key=lambda x: int(x) if x.isdigit() else x)
            
            # Optionnel: trier les mots alphabétiquement pour faciliter la recherche
            all_words_sorted = sorted(all_words)
            
            for w in all_words_sorted:
                row = {"Word": w}
                for c in classes:
                    row[f"P(w|{c})"] = f"{st.session_state.cond_probs[w].get(c, 0):.8f}"
                cond_data.append(row)
            
            cond_df = pd.DataFrame(cond_data)
            
            # Recherche dans les probabilités conditionnelles
            search_cond = st.text_input("Rechercher un mot:", key="search_cond")
            if search_cond:
                cond_df = cond_df[cond_df["Word"].str.contains(search_cond, case=False, na=False)]
            
            st.dataframe(cond_df, use_container_width=True, height=400)
            
            # Bouton d'export
            if st.button("💾 Exporter les probabilités conditionnelles (CSV)"):
                csv = cond_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Télécharger CSV",
                    data=csv,
                    file_name="conditional_probabilities.csv",
                    mime="text/csv"
                )

# Afficher les résultats des tests
if 'test_results' in st.session_state and st.session_state.test_results:
    st.markdown("---")
    st.header("🧪 Résultats des tests")
    
    # Afficher la précision
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Précision", f"{st.session_state.accuracy:.2%}")
    with col2:
        total_tests = len(st.session_state.test_results)
        correct_tests = sum(1 for r in st.session_state.test_results if r["Correct"] == "✓")
        st.metric("Corrects", f"{correct_tests}/{total_tests}")
    with col3:
        accuracy_per_class = {}
        test_df = pd.DataFrame(st.session_state.test_results)
        if not test_df.empty:
            for label_num in ["1", "2", "3", "4"]:
                class_data = test_df[test_df["Vrai Num"] == label_num]
                if len(class_data) > 0:
                    accuracy = len(class_data[class_data["Correct"] == "✓"]) / len(class_data)
                    accuracy_per_class[label_num] = accuracy
        
        best_class = max(accuracy_per_class.items(), key=lambda x: x[1])[0] if accuracy_per_class else "N/A"
        st.metric("Meilleure classe", LABEL_MAP.get(best_class, best_class))
    
    # Recherche et tri dans les résultats de test
    st.subheader("Détail des résultats")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        search_test = st.text_input("Rechercher dans les tests:", placeholder="Article, label ou base...")
    with col2:
        sort_test_by = st.selectbox("Trier par:", ["Article", "Correct", "Vrai Label", "Label Prédit"], key="sort_test")
    with col3:
        filter_correct = st.selectbox("Filtrer par résultat:", ["Tous", "Corrects (✓)", "Incorrects (✗)"])
    
    # Préparer les données pour l'affichage
    results_df = pd.DataFrame(st.session_state.test_results)
    
    if not results_df.empty:
        # Convertir Article en numérique pour un tri correct
        results_df["Article_Num"] = pd.to_numeric(results_df["Article"], errors='coerce')
        
        # Filtrer par recherche
        if search_test:
            mask = results_df.apply(lambda row: row.astype(str).str.contains(search_test, case=False).any(), axis=1)
            results_df = results_df[mask]
        
        # Filtrer par résultat
        if filter_correct == "Corrects (✓)":
            results_df = results_df[results_df["Correct"] == "✓"]
        elif filter_correct == "Incorrects (✗)":
            results_df = results_df[results_df["Correct"] == "✗"]
        
        # Trier
        if sort_test_by == "Article":
            results_df = results_df.sort_values("Article_Num")
        else:
            results_df = results_df.sort_values(sort_test_by)
        
        # Afficher le tableau
        display_df = results_df[["Article", "Vrai Label", "Label Prédit", "Correct", "Base"]].reset_index(drop=True)
        
        # Colorer les lignes
        def color_correct(val):
            color = 'background-color: #90EE90' if val == '✓' else 'background-color: #FFCCCB'
            return color
        
        st.dataframe(
            display_df.style.applymap(color_correct, subset=['Correct']),
            use_container_width=True,
            height=400
        )
        
        # Matrice de confusion
        st.subheader("📊 Matrice de confusion")
        
        confusion_data = []
        true_labels = sorted(results_df["Vrai Num"].unique())
        pred_labels = sorted(results_df["Prédit Num"].unique())
        
        # Créer la matrice
        for true in true_labels:
            row = {"Vrai": LABEL_MAP.get(true, true)}
            for pred in pred_labels:
                count = len(results_df[(results_df["Vrai Num"] == true) & (results_df["Prédit Num"] == pred)])
                row[LABEL_MAP.get(pred, pred)] = count
            confusion_data.append(row)
        
        confusion_df = pd.DataFrame(confusion_data)
        confusion_df = confusion_df.set_index("Vrai")
        
        # Afficher la matrice
        st.dataframe(confusion_df, use_container_width=True)
        
        # Statistiques par classe
        st.subheader("📈 Statistiques par classe")
        
        stats_data = []
        for label_num in ["1", "2", "3", "4"]:
            class_data = results_df[results_df["Vrai Num"] == label_num]
            if len(class_data) > 0:
                correct = len(class_data[class_data["Correct"] == "✓"])
                total = len(class_data)
                accuracy = correct / total if total > 0 else 0
                
                stats_data.append({
                    "Classe": label_num,
                    "Label": LABEL_MAP.get(label_num, "Unknown"),
                    "Total": total,
                    "Corrects": correct,
                    "Précision": f"{accuracy:.2%}"
                })
        
        stats_df = pd.DataFrame(stats_data)
        st.dataframe(stats_df, use_container_width=True)

# Page d'accueil si pas de données
if 'training_data' not in st.session_state or not st.session_state.training_data:
    st.markdown("""
    ## 📋 Instructions d'utilisation
    
    ### **Objectif:**
    Classificateur Naive Bayes pour catégoriser des articles scientifiques en 4 classes.
    
    ### **Classes:**
    1. **Metaheuristics** (Métaheuristiques)
    2. **Machine & Deep Learning** (Apprentissage automatique et profond)
    3. **Combination of Metaheuristics & Machine/Deep Learning** (Combinaison)
    4. **Others** (Autres)
    
    ### **Modes d'entraînement disponibles:**
    
    #### **📁 Volume unique** (mode original):
    ```
    dossier_articles/                    dossier_labels/
    ├── Article_1.txt                    ├── Article_1_label.txt  (contenu: "1")
    ├── Article_2.txt                    ├── Article_2_label.txt  (contenu: "2")
    └── Article_3.txt                    └── Article_3_label.txt  (contenu: "3")
    ```
    
    #### **📚 Multiples volumes** (nouveau mode):
    ```
    dossier_volumes/
    ├── Volume_1/
    │   ├── Article_1.txt
    │   ├── Article_1_label.txt  (contenu: "1")
    │   ├── Article_2.txt
    │   └── Article_2_label.txt  (contenu: "2")
    ├── Volume_2/
    │   ├── Article_3.txt
    │   ├── Article_3_label.txt  (contenu: "3")
    │   ├── Article_4.txt
    │   └── Article_4_label.txt  (contenu: "4")
    └── Volume_3/
        ├── Article_5.txt
        └── Article_5_label.txt  (contenu: "1")
    ```
    
    #### **Pour le test** (identique pour les deux modes):
    ```
    dossier_test/
    ├── Article_101.txt
    ├── Article_101_label.txt
    ├── Article_102.txt
    └── Article_102_label.txt
    ```
    
    ### **Étapes:**
    1. Sélectionnez le mode d'entraînement ("Volume unique" ou "Multiples volumes")
    2. Sélectionnez le mode "Entraînement + Test"
    3. Entrez les chemins selon le mode choisi:
       - **Volume unique:** chemins des dossiers articles et labels (séparés)
       - **Multiples volumes:** chemin du dossier racine contenant les volumes
    4. Cliquez sur "Charger l'entraînement"
    5. Cliquez sur "Entraîner le modèle"
    6. Entrez le chemin du dossier de test
    7. Cliquez sur "Charger & Tester"
    
    ### **Algorithme (identique à l'original):**
    - **Prétraitement:** suppression stopwords, tokenization, stemming optionnel
    - **Priors:** P(c) = Nc / N
    - **Conditionnelles:** P(w|c) = (count(w,c) + 1) / (sum_words(c) + V)
    - **Prédiction:** argmax_c [ log(P(c)) + Σ log(P(w|c)) ]
    
    ### **Fonctionnalités d'affichage:**
    - ✅ Recherche dans les tableaux
    - ✅ Tri des articles (numérique)
    - ✅ Filtrage par résultat
    - ✅ Matrice de confusion
    - ✅ Statistiques par classe
    - ✅ Colonne "Volume" dans l'affichage
    - ✅ Statistiques par volume (mode multi-volumes)
    """)

# Note sur les dépendances
st.sidebar.markdown("---")
st.sidebar.markdown("### 📦 Dépendances nécessaires")
st.sidebar.code("""
pip install streamlit nltk pandas
""")