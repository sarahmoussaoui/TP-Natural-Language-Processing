import os
import re
import streamlit as st
import pandas as pd
import numpy as np
import glob
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# ===============================================
# Configuration et initialisation
# ===============================================
st.set_page_config(page_title="Testing - Modèle Logistic Regression", layout="wide")

# Mapping des classes (doit être identique à l'entraînement)
CLASS_MAPPING = {
    "1": "Metaheuristics",
    "2": "Machine & Deep Learning", 
    "3": "Combination of Metaheuristics & Machine/Deep Learning",
    "4": "Others"
}

# Classe du modèle (identique à l'entraînement)
class LogisticRegressionPyTorch(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(LogisticRegressionPyTorch, self).__init__()
        self.linear = nn.Linear(input_dim, num_classes)
        
    def forward(self, x):
        return self.linear(x)

# ===============================================
# Fonctions de chargement du modèle
# ===============================================
def load_saved_model(model_path):
    """Charge le modèle sauvegardé"""
    try:
        if not os.path.exists(model_path):
            st.error(f"❌ Fichier modèle non trouvé: {model_path}")
            return None, None, None, None, None
        
        st.info(f"Chargement du modèle: {model_path}")
        
        # Charger les données sauvegardées
        saved_data = torch.load(model_path, map_location=torch.device('cpu'), weights_only=False)
        
        # Reconstruire le modèle
        model = saved_data['model_class'](**saved_data['model_params'])
        model.load_state_dict(saved_data['model_state_dict'])
        model.eval()  # Mode évaluation
        
        # Récupérer les autres composants
        scaler = saved_data['scaler']
        attributes = saved_data['attributes']
        normalization_params = saved_data.get('normalization_params', {
            'lowercase': True,
            'remove_punctuation': True,
            'remove_numbers': False
        })
        
        class_mapping = saved_data.get('class_mapping', CLASS_MAPPING)
        
        st.success(f"""
        ✅ Modèle chargé avec succès!
        - Features: {model.linear.in_features}
        - Classes: {model.linear.out_features}
        - Attributs: {len(attributes)}
        """)
        
        return model, scaler, attributes, normalization_params, class_mapping
        
    except Exception as e:
        st.error(f"❌ Erreur lors du chargement du modèle: {str(e)}")
        return None, None, None, None, None

# ===============================================
# Fonctions de traitement de texte (identiques à l'entraînement)
# ===============================================
def normalize_text(text, lowercase=True, remove_punctuation=True, remove_numbers=True):
    """Normalise le texte selon les options choisies"""
    if lowercase:
        text = text.lower()
    
    if remove_punctuation:
        text = re.sub(r'[^\w\s-]', ' ', text)
    
    if remove_numbers:
        text = re.sub(r'\d+', '', text)
    
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def count_keyword_occurrences(text, keywords, normalization_params):
    """Compte le nombre d'occurrences EXACTES des keywords dans le texte"""
    normalized_text = normalize_text(text, **normalization_params)
    count = 0
    
    # Tokenization pour séparer les mots
    tokens = re.findall(r'\b[\w-]+\b', normalized_text)
    
    for keyword in keywords:
        normalized_keyword = normalize_text(keyword, **normalization_params)
        
        if normalized_keyword:
            if ' ' not in normalized_keyword:
                for token in tokens:
                    if token == normalized_keyword:
                        count += 1
            else:
                pattern = r'\b' + re.escape(normalized_keyword) + r'\b'
                matches = len(re.findall(pattern, normalized_text))
                count += matches
    
    return count

def create_feature_vector(text, attributes, normalization_params):
    """Crée un vecteur de features pour un texte donné"""
    features = {}
    for attr_name, keywords in attributes.items():
        count = count_keyword_occurrences(text, keywords, normalization_params)
        features[attr_name] = count
    
    # Convertir en liste dans le bon ordre
    feature_vector = [features.get(attr_name, 0) for attr_name in attributes.keys()]
    return np.array(feature_vector).reshape(1, -1)

# ===============================================
# Fonctions de prédiction
# ===============================================
def predict_single_article(model, scaler, text, attributes, normalization_params):
    """Prédit la classe d'un article"""
    # Créer le vecteur de features
    feature_vector = create_feature_vector(text, attributes, normalization_params)
    
    # Normaliser
    feature_vector_scaled = scaler.transform(feature_vector)
    
    # Convertir en tenseur
    X_tensor = torch.FloatTensor(feature_vector_scaled)
    
    # Prédiction
    with torch.no_grad():
        outputs = model(X_tensor)
        _, predicted = torch.max(outputs.data, 1)
    
    # Convertir en label 1-index
    predicted_class = predicted.numpy()[0] + 1
    
    return predicted_class, feature_vector_scaled

def extract_label_from_file(label_file_path):
    """Extrait le label depuis un fichier label"""
    try:
        with open(label_file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read().strip()
            # Chercher un chiffre entre 1 et 4
            match = re.search(r'[1-4]', content)
            if match:
                return match.group(0)
            else:
                # Essayer une autre méthode - vérifier le premier caractère non vide
                content_clean = content.strip()
                if content_clean and content_clean[0] in ['1', '2', '3', '4']:
                    return content_clean[0]
                else:
                    st.warning(f"Contenu label non valide dans {label_file_path}: '{content}'")
                    return "?"
    except Exception as e:
        st.warning(f"Erreur lecture label {label_file_path}: {str(e)}")
        return "?"

# ===============================================
# Fonctions de chargement des données de test - VERSION CORRIGÉE
# ===============================================
def extract_article_name(filename):
    """
    Extrait le nom de l'article depuis un filename.
    Gère les cas: Article_1, Article_1.txt, Article_1_label.txt
    """
    # Enlever l'extension si elle existe
    base_name = os.path.splitext(filename)[0]
    
    # Enlever "_label" s'il existe
    if base_name.endswith('_label'):
        base_name = base_name[:-6]  # Enlève "_label"
    
    return base_name

def load_test_articles_from_folder(folder_path):
    """Charge tous les articles d'un dossier (structure simple)"""
    articles_data = []
    
    if not os.path.exists(folder_path):
        st.error(f"❌ Dossier non trouvé: {folder_path}")
        return []
    
    # Dictionnaire pour stocker tous les fichiers trouvés
    all_files = {}
    
    # Parcourir tous les fichiers dans le dossier
    for file_path in glob.glob(os.path.join(folder_path, "*")):
        if os.path.isfile(file_path):
            filename = os.path.basename(file_path)
            
            # Ignorer les fichiers cachés
            if filename.startswith('.'):
                continue
            
            # Extraire le nom de l'article
            article_name = extract_article_name(filename)
            
            if article_name not in all_files:
                all_files[article_name] = {
                    'article_file': None,
                    'label_file': None
                }
            
            # Identifier le type de fichier
            if "_label" in filename or filename.endswith('_label'):
                all_files[article_name]['label_file'] = file_path
            else:
                all_files[article_name]['article_file'] = file_path
    
    # Afficher les fichiers trouvés pour débogage
    st.info(f"📁 {len(all_files)} articles potentiels trouvés dans le dossier")
    
    # Traiter chaque article
    for article_name, files in all_files.items():
        article_file = files.get('article_file')
        label_file = files.get('label_file')
        
        # S'il n'y a pas de fichier article, passer
        if not article_file:
            continue
        
        try:
            # Charger le contenu de l'article
            with open(article_file, 'r', encoding='utf-8', errors='ignore') as f:
                text = f.read()
            
            # Vérifier si le fichier n'est pas vide
            if not text.strip():
                st.warning(f"⚠️ Fichier article vide: {os.path.basename(article_file)}")
                continue
            
            # Extraire le label
            true_label = "?"
            if label_file and os.path.exists(label_file):
                true_label = extract_label_from_file(label_file)
                if true_label == "?":
                    st.warning(f"⚠️ Label non valide ou fichier vide: {os.path.basename(label_file)}")
            else:
                st.warning(f"⚠️ Fichier label manquant pour: {os.path.basename(article_file)}")
            
            articles_data.append({
                'volume': 'Dossier unique',
                'filename': os.path.basename(article_file),
                'article_name': article_name,
                'text': text,
                'true_label': true_label,
                'article_path': article_file,
                'label_path': label_file if label_file else "Non trouvé",
                'text_length': len(text)
            })
            
        except Exception as e:
            st.error(f"❌ Erreur lors du chargement de {os.path.basename(article_file)}: {str(e)}")
    
    return articles_data

def load_test_articles_from_volume_structure(main_folder_path):
    """Charge les articles depuis une structure de volumes (dossiers dans un dossier principal)"""
    articles_data = []
    
    if not os.path.exists(main_folder_path):
        st.error(f"❌ Dossier principal non trouvé: {main_folder_path}")
        return []
    
    # Parcourir tous les dossiers (volumes)
    for volume_name in sorted(os.listdir(main_folder_path)):
        volume_path = os.path.join(main_folder_path, volume_name)
        
        if not os.path.isdir(volume_path):
            continue
        
        # Dictionnaire pour stocker tous les fichiers trouvés dans ce volume
        all_files = {}
        
        # Parcourir tous les fichiers dans le volume
        for file_path in glob.glob(os.path.join(volume_path, "*")):
            if os.path.isfile(file_path):
                filename = os.path.basename(file_path)
                
                # Ignorer les fichiers cachés
                if filename.startswith('.'):
                    continue
                
                # Extraire le nom de l'article
                article_name = extract_article_name(filename)
                
                if article_name not in all_files:
                    all_files[article_name] = {
                        'article_file': None,
                        'label_file': None
                    }
                
                # Identifier le type de fichier
                if "_label" in filename or filename.endswith('_label'):
                    all_files[article_name]['label_file'] = file_path
                else:
                    all_files[article_name]['article_file'] = file_path
        
        st.info(f"📁 Volume '{volume_name}': {len(all_files)} articles potentiels trouvés")
        
        # Traiter chaque article dans ce volume
        for article_name, files in all_files.items():
            article_file = files.get('article_file')
            label_file = files.get('label_file')
            
            # S'il n'y a pas de fichier article, passer
            if not article_file:
                continue
            
            try:
                # Charger le contenu de l'article
                with open(article_file, 'r', encoding='utf-8', errors='ignore') as f:
                    text = f.read()
                
                # Vérifier si le fichier n'est pas vide
                if not text.strip():
                    st.warning(f"⚠️ Fichier article vide: {volume_name}/{os.path.basename(article_file)}")
                    continue
                
                # Extraire le label
                true_label = "?"
                if label_file and os.path.exists(label_file):
                    true_label = extract_label_from_file(label_file)
                    if true_label == "?":
                        st.warning(f"⚠️ Label non valide ou fichier vide: {volume_name}/{os.path.basename(label_file)}")
                else:
                    st.warning(f"⚠️ Fichier label manquant pour: {volume_name}/{os.path.basename(article_file)}")
                
                articles_data.append({
                    'volume': volume_name,
                    'filename': os.path.basename(article_file),
                    'article_name': article_name,
                    'text': text,
                    'true_label': true_label,
                    'article_path': article_file,
                    'label_path': label_file if label_file else "Non trouvé",
                    'text_length': len(text)
                })
                
            except Exception as e:
                st.error(f"❌ Erreur lors du chargement de {volume_name}/{os.path.basename(article_file)}: {str(e)}")
    
    return articles_data

# ===============================================
# Interface Streamlit
# ===============================================
st.title("🧪 Testing - Modèle Logistic Regression")
st.markdown("""
**Master 2 SII - Module TALN - Université USTHB 2025/2026**  
*Testing du modèle de régression logistique sur de nouveaux articles*
""")

st.markdown("---")

# Sidebar pour la configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Chargement du modèle
    st.subheader("Chargement du modèle")
    model_path = st.text_input(
        "Chemin du modèle sauvegardé:",
        value="logistic_regression_model.pth",
        help="Fichier .pth contenant le modèle entraîné"
    )
    
    if st.button("📂 Charger le modèle", type="primary", use_container_width=True):
        with st.spinner("Chargement du modèle..."):
            (st.session_state.model, 
             st.session_state.scaler, 
             st.session_state.attributes, 
             st.session_state.normalization_params,
             st.session_state.class_mapping) = load_saved_model(model_path)
    
    # Sélection du mode de test
    st.subheader("Mode de testing")
    test_mode = st.selectbox(
        "Structure des données de test:",
        [
            "📁 Dossier unique (tous les articles dans un dossier)",
            "🗂️ Structure volumique (dossiers de volumes avec articles)"
        ]
    )
    
    # Chemins selon le mode
    if test_mode == "📁 Dossier unique (tous les articles dans un dossier)":
        test_data_path = st.text_input(
            "Chemin du dossier de test:",
            placeholder="C:/chemin/vers/dossier_test/",
            help="Dossier contenant les articles (Article_XX.txt ou Article_XX) et labels (Article_XX_label.txt ou Article_XX_label)"
        )
    else:  # Structure volumique
        test_data_path = st.text_input(
            "Chemin du dossier principal:",
            placeholder="C:/chemin/vers/dossier_principal/",
            help="Dossier contenant des sous-dossiers (volumes) avec articles"
        )
    
    # Options de débogage
    show_debug_info = st.checkbox("Afficher les informations de débogage", value=False)
    
    # Bouton pour charger les données de test
    if st.button("📥 Charger les articles de test", type="secondary", use_container_width=True):
        if not test_data_path or not os.path.exists(test_data_path):
            st.error("❌ Chemin des données de test invalide!")
        else:
            with st.spinner("Chargement des articles de test..."):
                if test_mode == "📁 Dossier unique (tous les articles dans un dossier)":
                    articles_data = load_test_articles_from_folder(test_data_path)
                else:
                    articles_data = load_test_articles_from_volume_structure(test_data_path)
                
                if articles_data:
                    st.session_state.test_articles = articles_data
                    st.success(f"✅ {len(articles_data)} articles chargés")
                    
                    # Afficher les informations de débogage si demandé
                    if show_debug_info:
                        with st.expander("🔍 Informations de débogage"):
                            debug_info = []
                            for article in articles_data[:10]:  # Afficher seulement les 10 premiers
                                debug_info.append({
                                    'Nom article': article['article_name'],
                                    'Fichier article': article['filename'],
                                    'Fichier label': os.path.basename(article['label_path']) if article['label_path'] != "Non trouvé" else "Non trouvé",
                                    'Label extrait': article['true_label']
                                })
                            if debug_info:
                                st.dataframe(pd.DataFrame(debug_info), use_container_width=True)
                else:
                    st.error("❌ Aucun article trouvé dans le chemin spécifié")

# Zone principale
if 'model' not in st.session_state:
    st.session_state.model = None
if 'test_articles' not in st.session_state:
    st.session_state.test_articles = []

# Section 1: Information du modèle
st.header("📋 Information du modèle")

if st.session_state.model is not None:
    model = st.session_state.model
    st.success(f"""
    **Modèle chargé:**
    - **Features d'entrée:** {model.linear.in_features}
    - **Classes de sortie:** {model.linear.out_features}
    - **Nombre d'attributs:** {len(st.session_state.attributes) if st.session_state.attributes else 'N/A'}
    """)
    
    # Afficher les attributs
    with st.expander("👁️ Voir les attributs chargés"):
        if st.session_state.attributes:
            attributs_df = pd.DataFrame({
                'Attribut': list(st.session_state.attributes.keys()),
                'Nombre de mots': [len(keywords) for keywords in st.session_state.attributes.values()]
            })
            st.dataframe(attributs_df, use_container_width=True)
else:
    st.warning("⚠️ Chargez d'abord un modèle dans la sidebar")

# Section 2: Articles chargés
if st.session_state.test_articles:
    st.markdown("---")
    st.header("📚 Articles de test chargés")
    
    # Afficher la liste des articles
    articles_df = pd.DataFrame(st.session_state.test_articles)
    st.info(f"**{len(articles_df)} articles chargés**")
    
    # Statistiques détaillées
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        articles_with_label = sum(1 for x in articles_df['true_label'] if x in ['1', '2', '3', '4'])
        st.metric("Articles avec label", f"{articles_with_label}/{len(articles_df)}")
    with col2:
        articles_without_label = sum(1 for x in articles_df['true_label'] if x == "?")
        st.metric("Articles sans label", f"{articles_without_label}")
    with col3:
        # Détecter les extensions utilisées
        extensions = set()
        for filename in articles_df['filename']:
            if '.' in filename:
                extensions.add(os.path.splitext(filename)[1])
            else:
                extensions.add("(sans extension)")
        st.metric("Extensions", f"{len(extensions)} type(s)")
    with col4:
        if articles_with_label > 0:
            label_dist = articles_df[articles_df['true_label'].isin(['1', '2', '3', '4'])]['true_label'].value_counts()
            most_common_label = label_dist.index[0] if len(label_dist) > 0 else "N/A"
            st.metric("Label le + fréquent", CLASS_MAPPING.get(most_common_label, most_common_label))
        else:
            st.metric("Label le + fréquent", "N/A")
    
    # Afficher le tableau des articles
    st.subheader("📋 Liste des articles")
    
    display_df = articles_df.copy()
    if test_mode == "📁 Dossier unique (tous les articles dans un dossier)":
        display_cols = ['filename', 'true_label', 'article_name']
    else:
        display_cols = ['volume', 'filename', 'true_label', 'article_name']
    
    display_df = display_df[display_cols]
    display_df.columns = ['Fichier', 'Label réel', 'Nom article'] if test_mode == "📁 Dossier unique (tous les articles dans un dossier)" else ['Volume', 'Fichier', 'Label réel', 'Nom article']
    
    # Colorer les labels
    def color_label(val):
        if val == "?":
            return 'background-color: #FFCCCB'
        elif val in ['1', '2', '3', '4']:
            return 'background-color: #90EE90'
        return ''
    
    st.dataframe(
        display_df.style.applymap(color_label, subset=['Label réel']),
        use_container_width=True,
        height=400
    )
    
    # Distribution des labels réels
    st.subheader("📊 Distribution des labels réels")
    
    # Créer un graphique
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Graphique 1: Avec et sans label
    labels_present = ['Avec label', 'Sans label']
    counts = [articles_with_label, articles_without_label]
    colors = ['#4CAF50', '#F44336']
    ax1.bar(labels_present, counts, color=colors)
    ax1.set_title('Articles avec/sans label')
    ax1.set_ylabel('Nombre d\'articles')
    
    # Ajouter les nombres sur les barres
    for i, (label, count) in enumerate(zip(labels_present, counts)):
        ax1.text(i, count + 0.5, str(count), ha='center', va='bottom')
    
    # Graphique 2: Distribution des labels
    if articles_with_label > 0:
        label_counts = articles_df[articles_df['true_label'].isin(['1', '2', '3', '4'])]['true_label'].value_counts().sort_index()
        labels = [CLASS_MAPPING.get(str(k), f"Classe {k}") for k in label_counts.index]
        
        bars = ax2.bar(labels, label_counts.values, color='#2196F3')
        ax2.set_title('Distribution des labels')
        ax2.set_xlabel('Classe')
        ax2.set_ylabel('Nombre d\'articles')
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # Ajouter les nombres sur les barres
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                   f'{int(height)}', ha='center', va='bottom')
    else:
        ax2.text(0.5, 0.5, 'Aucun label trouvé', 
                ha='center', va='center', transform=ax2.transAxes, fontsize=12)
        ax2.set_title('Distribution des labels')
    
    plt.tight_layout()
    st.pyplot(fig)

# Section 3: Exécution des prédictions
if st.session_state.model is not None and st.session_state.test_articles:
    st.markdown("---")
    st.header("🎯 Exécution des prédictions")
    
    if st.button("🚀 Lancer les prédictions", type="primary", use_container_width=True):
        with st.spinner("Calcul des prédictions en cours..."):
            results = []
            all_features = []
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i, article in enumerate(st.session_state.test_articles):
                # Prédiction
                predicted_class, features = predict_single_article(
                    st.session_state.model,
                    st.session_state.scaler,
                    article['text'],
                    st.session_state.attributes,
                    st.session_state.normalization_params
                )
                
                # Vérifier si la prédiction est correcte
                if article['true_label'] in ['1', '2', '3', '4']:
                    correct = "✓" if str(predicted_class) == article['true_label'] else "✗"
                else:
                    correct = "N/A"
                
                results.append({
                    'Volume': article['volume'],
                    'Fichier': article['filename'],
                    'Label réel': article['true_label'],
                    'Classe réelle': CLASS_MAPPING.get(article['true_label'], f"Classe {article['true_label']}") if article['true_label'] != "?" else "Non labellisé",
                    'Label prédit': str(predicted_class),
                    'Classe prédite': CLASS_MAPPING.get(str(predicted_class), f"Classe {predicted_class}"),
                    'Correct': correct,
                    'Chemin': article['article_path']
                })
                
                all_features.append(features.flatten())
                
                # Mettre à jour la progression
                progress = (i + 1) / len(st.session_state.test_articles)
                progress_bar.progress(progress)
                status_text.text(f"Traitement: {i+1}/{len(st.session_state.test_articles)} articles")
            
            progress_bar.empty()
            status_text.empty()
            
            # Stocker les résultats
            st.session_state.prediction_results = results
            st.session_state.all_features = np.array(all_features)
            
            st.success(f"✅ Prédictions terminées sur {len(results)} articles!")
    
    # Affichage des résultats si disponibles
    if 'prediction_results' in st.session_state:
        results = st.session_state.prediction_results
        results_df = pd.DataFrame(results)
        
        st.markdown("---")
        st.header("📊 Résultats des prédictions")
        
        # Calcul des métriques (seulement pour les articles labellisés)
        labelled_results = [r for r in results if r['Label réel'] in ['1', '2', '3', '4']]
        if labelled_results:
            correct_count = sum(1 for r in labelled_results if r['Correct'] == '✓')
            total_count = len(labelled_results)
            accuracy = correct_count / total_count if total_count > 0 else 0
        else:
            correct_count = 0
            total_count = 0
            accuracy = 0
        
        # Métriques principales
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Accuracy", f"{accuracy:.2%}" if total_count > 0 else "N/A")
        with col2:
            st.metric("Corrects", f"{correct_count}/{total_count}" if total_count > 0 else "N/A")
        with col3:
            st.metric("Incorrects", f"{total_count - correct_count}/{total_count}" if total_count > 0 else "N/A")
        with col4:
            # Trouver la classe la mieux prédite
            if len(results) > 0:
                pred_counts = Counter([r['Label prédit'] for r in results])
                most_common = pred_counts.most_common(1)[0][0]
                st.metric("Classe la + prédite", CLASS_MAPPING.get(most_common, most_common))
        
        # Tableau des résultats détaillés
        st.subheader("📋 Détail des prédictions")
        
        # Options d'affichage
        col1, col2 = st.columns(2)
        with col1:
            show_all = st.checkbox("Afficher toutes les colonnes", value=False, key="show_all")
        with col2:
            filter_correct = st.selectbox("Filtrer par résultat:", 
                                         ["Tous", "Corrects seulement", "Incorrects seulement", "Non labellisés"],
                                         key="filter_correct")
        
        # Préparer le DataFrame d'affichage
        if test_mode == "📁 Dossier unique (tous les articles dans un dossier)":
            display_cols = ['Fichier', 'Classe réelle', 'Classe prédite', 'Correct']
        else:
            display_cols = ['Volume', 'Fichier', 'Classe réelle', 'Classe prédite', 'Correct']
        
        if show_all:
            display_cols = [col for col in results_df.columns if col not in ['Chemin', 'Label réel', 'Label prédit']]
        
        display_df = results_df[display_cols].copy()
        
        # Appliquer le filtre
        if filter_correct == "Corrects seulement":
            display_df = display_df[results_df['Correct'] == '✓']
        elif filter_correct == "Incorrects seulement":
            display_df = display_df[results_df['Correct'] == '✗']
        elif filter_correct == "Non labellisés":
            display_df = display_df[results_df['Correct'] == 'N/A']
        
        # Colorer les résultats
        def color_correct(val):
            if val == '✓':
                return 'background-color: #90EE90'
            elif val == '✗':
                return 'background-color: #FFCCCB'
            elif val == 'N/A':
                return 'background-color: #E0E0E0'
            return ''
        
        st.dataframe(
            display_df.style.applymap(color_correct, subset=['Correct']),
            use_container_width=True,
            height=400
        )
        
        # Matrice de confusion (seulement pour les articles labellisés)
        if labelled_results:
            st.subheader("📈 Matrice de confusion")
            
            # Extraire les labels réels et prédits
            y_true = [r['Label réel'] for r in labelled_results]
            y_pred = [r['Label prédit'] for r in labelled_results]
            
            if y_true and y_pred:
                cm = confusion_matrix(y_true, y_pred, labels=['1', '2', '3', '4'])
                
                cm_df = pd.DataFrame(
                    cm,
                    index=[f"Réel: {CLASS_MAPPING[str(i)]}" for i in range(1, 5)],
                    columns=[f"Prédit: {CLASS_MAPPING[str(i)]}" for i in range(1, 5)]
                )
                
                # Afficher avec coloration
                st.dataframe(
                    cm_df.style.background_gradient(cmap='Blues', axis=None),
                    use_container_width=True
                )
                
                # Rapport de classification
                st.subheader("📊 Rapport de classification")
                
                try:
                    report = classification_report(
                        y_true, y_pred,
                        target_names=[CLASS_MAPPING[str(i)] for i in range(1, 5)],
                        output_dict=True
                    )
                    
                    report_df = pd.DataFrame(report).transpose()
                    st.dataframe(report_df.style.format("{:.3f}"), use_container_width=True)
                except:
                    st.warning("Impossible de générer le rapport de classification")
        
        # Export des résultats
        st.subheader("💾 Export des résultats")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Export CSV
            csv = results_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Télécharger les résultats (CSV)",
                data=csv,
                file_name="resultats_prediction.csv",
                mime="text/csv"
            )
        
        with col2:
            # Export détaillé avec features
            if 'all_features' in st.session_state:
                detailed_df = results_df.copy()
                for i in range(st.session_state.all_features.shape[1]):
                    detailed_df[f'Feature_{i+1}'] = st.session_state.all_features[:, i]
                
                csv_detailed = detailed_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Télécharger résultats détaillés",
                    data=csv_detailed,
                    file_name="resultats_detailles.csv",
                    mime="text/csv"
                )
        
        # Visualisation des features
        st.subheader("🔍 Visualisation des features")
        
        if 'all_features' in st.session_state and st.session_state.all_features.shape[1] > 0:
            # Moyenne des features par classe prédite
            features_by_class = {}
            for class_num in ['1', '2', '3', '4']:
                class_indices = [i for i, r in enumerate(results) if r['Label prédit'] == class_num]
                if class_indices:
                    features_by_class[CLASS_MAPPING[class_num]] = st.session_state.all_features[class_indices].mean(axis=0)
            
            if features_by_class:
                features_df = pd.DataFrame(features_by_class)
                features_df.index = [f"A{i+1}" for i in range(features_df.shape[0])]
                
                fig, ax = plt.subplots(figsize=(12, 6))
                features_df.plot(kind='bar', ax=ax)
                ax.set_title('Moyenne des features par classe prédite')
                ax.set_xlabel('Attributs')
                ax.set_ylabel('Valeur moyenne')
                ax.legend(title='Classe')
                plt.xticks(rotation=45, ha='right')
                st.pyplot(fig)

# Instructions
with st.expander("📚 Instructions d'utilisation"):
    st.markdown("""
    ### **Étapes pour le testing:**
    
    1. **Charger le modèle**
       - Entrez le chemin du fichier `.pth` sauvegardé
       - Cliquez sur "Charger le modèle"
    
    2. **Charger les données de test**
       - Sélectionnez la structure de vos données
       - Entrez le chemin approprié
       - Cliquez sur "Charger les articles de test"
    
    3. **Lancer les prédictions**
       - Cliquez sur "Lancer les prédictions"
       - Attendez le traitement
    
    4. **Analyser les résultats**
       - Consultez l'accuracy et les métriques
       - Vérifiez la matrice de confusion
       - Téléchargez les résultats
    
    ### **Structure des fichiers acceptée:**
    
    #### **Articles (accepte avec ou sans extension):**
    ```
    Article_1.txt    ou    Article_1    (sans extension)
    Article_2.txt    ou    Article_2    (sans extension)
    ```
    
    #### **Labels (doivent contenir _label):**
    ```
    Article_1_label.txt    ou    Article_1_label    (sans extension)
    Article_2_label.txt    ou    Article_2_label    (sans extension)
    ```
    
    #### **Contenu des fichiers label:**
    ```
    1  # pour Metaheuristics
    2  # pour Machine & Deep Learning  
    3  # pour Combination of Metaheuristics & Machine/Deep Learning
    4  # pour Others
    ```
    
    ### **Notes importantes:**
    - Les articles sans fichiers label seront marqués comme "Non labellisé"
    - Les fichiers peuvent avoir l'extension `.txt` ou aucune extension
    - Le modèle doit avoir été entraîné avec les mêmes attributs
    - Activez "Afficher les informations de débogage" pour voir comment les fichiers sont associés
    """)

# Installation requirements
st.sidebar.markdown("---")
st.sidebar.markdown("### 📦 Dépendances nécessaires")
st.sidebar.code("""
pip install streamlit pandas numpy torch matplotlib scikit-learn
""")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666; font-size: 0.9em;'>
    Master 2 SII - Module TALN - Université USTHB 2025/2026<br>
    Application de testing pour modèle de régression logistique
    </div>
    """,
    unsafe_allow_html=True
)