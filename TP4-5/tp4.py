import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import os
import re
from collections import defaultdict, Counter
import math
from nltk.stem import PorterStemmer
import nltk
from nltk.corpus import stopwords


class NaiveBayesClassifier:
    def __init__(self):
        # Obtenir le chemin absolu du script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.articles_dir = os.path.join(script_dir, "TP", "All-in-many")
        self.labels_dir = os.path.join(script_dir, "TP", "All-in-many_classification")
        self.volume_number = 18
        
        # Données
        self.articles = {}  # {article_num: text}
        self.labels = {}    # {article_num: class_label}
        self.classes = set()
        
        # Probabilités
        self.class_probs = {}  # P(c)
        self.word_given_class_probs = defaultdict(lambda: defaultdict(float))  # P(w|c)
        self.vocabulary = set()
        
        # Options
        self.use_stemming = False
        self.use_normalization = False
        self.stemmer = PorterStemmer()
        
        # Stop words
        self.stop_words = set(stopwords.words('english'))
        
    def load_data(self, volume=18):
        """Charge les articles et leurs labels pour un volume donné"""
        self.volume_number = volume
        self.articles.clear()
        self.labels.clear()
        self.classes.clear()
        
        # Expression régulière pour extraire le numéro d'article
        pattern = re.compile(rf"Article_(\d+)_Volume_{volume}$")
        
        # Charger les articles
        articles_path = os.path.join(self.articles_dir)
        if os.path.exists(articles_path):
            for filename in os.listdir(articles_path):
                match = pattern.match(filename)
                if match:
                    article_num = int(match.group(1))
                    filepath = os.path.join(articles_path, filename)
                    with open(filepath, 'r', encoding='utf-8') as f:
                        self.articles[article_num] = f.read().strip()
        
        # Charger les labels
        labels_path = os.path.join(self.labels_dir)
        if os.path.exists(labels_path):
            for filename in os.listdir(labels_path):
                match = pattern.match(filename)
                if match:
                    article_num = int(match.group(1))
                    filepath = os.path.join(labels_path, filename)
                    with open(filepath, 'r', encoding='utf-8') as f:
                        label = int(f.read().strip())
                        self.labels[article_num] = label
                        self.classes.add(label)
        
        return len(self.articles), len(self.classes)
    
    def preprocess_text(self, text):
        """Prétraite le texte : tokenisation, normalisation, stemming"""
        # Nettoyer la ponctuation simple
        text = re.sub(r'[^\w\s]', ' ', text)
        
        # Tokenisation simple avec split()
        words = text.lower().split()
        
        # Supprimer les stop words
        words = [word for word in words if word not in self.stop_words]
        
        # Ne pas appliquer de stemming par défaut
        # (le stemming sera désactivé dans l'interface)
        if self.use_stemming:
            words = [self.stemmer.stem(word) for word in words]
        
        return words
    
    def train(self):
        """Entraîne le classificateur Naive Bayes"""
        # Compter les documents par classe
        class_counts = Counter(self.labels.values())
        total_docs = len(self.labels)
        
        # Calculer P(c)
        self.class_probs = {c: count / total_docs for c, count in class_counts.items()}
        
        # Construire le vocabulaire et compter les mots par classe
        word_counts_by_class = defaultdict(Counter)
        total_words_by_class = defaultdict(int)
        
        for article_num, label in self.labels.items():
            if article_num in self.articles:
                words = self.preprocess_text(self.articles[article_num])
                self.vocabulary.update(words)
                word_counts_by_class[label].update(words)
                total_words_by_class[label] += len(words)
        
        vocab_size = len(self.vocabulary)
        
        # Calculer P(w|c) avec lissage de Laplace
        for c in self.classes:
            for word in self.vocabulary:
                count = word_counts_by_class[c][word]
                if self.use_normalization:
                    # Normalisation : lissage de Laplace
                    self.word_given_class_probs[word][c] = (count + 1) / (total_words_by_class[c] + vocab_size)
                else:
                    # Sans normalisation : lissage de Laplace
                    self.word_given_class_probs[word][c] = (count + 1) / (total_words_by_class[c] + vocab_size)
        
        return self.class_probs, dict(self.word_given_class_probs)
    
    def predict(self, article_num):
        """Prédit la classe d'un article"""
        if article_num not in self.articles:
            return None, {}
        
        text = self.articles[article_num]
        words = self.preprocess_text(text)
        
        # Calculer le score pour chaque classe (log-probabilités)
        scores = {}
        for c in self.classes:
            score = math.log(self.class_probs[c])
            for word in words:
                if word in self.word_given_class_probs:
                    prob = self.word_given_class_probs[word][c]
                    if prob > 0:
                        score += math.log(prob)
            scores[c] = score
        
        # Trouver la classe avec le score maximal
        predicted_class = max(scores, key=scores.get)
        
        return predicted_class, scores
    
    def get_article_info(self, article_num):
        """Retourne les informations d'un article"""
        if article_num in self.articles:
            return {
                'article_num': article_num,
                'label': self.labels.get(article_num, "N/A"),
                'text': self.articles[article_num]
            }
        return None


class NaiveBayesGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Part 5 - Text Classification - Naive Bayes")
        self.root.geometry("1400x900")
        
        self.classifier = NaiveBayesClassifier()
        
        self.create_widgets()
        
    def create_widgets(self):
        # Frame principal
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configuration de la grille
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        
        # ===== Section 1: Configuration =====
        config_frame = ttk.LabelFrame(main_frame, text="Configuration", padding="10")
        config_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        
        # Volume selection
        ttk.Label(config_frame, text="Volume n°:").grid(row=0, column=0, padx=5)
        self.volume_var = tk.IntVar(value=18)
        volume_spinbox = ttk.Spinbox(config_frame, from_=1, to=100, textvariable=self.volume_var, width=10)
        volume_spinbox.grid(row=0, column=1, padx=5)
        
        # Mode selection
        self.mode_var = tk.StringVar(value="volume")
        ttk.Radiobutton(config_frame, text="Content of one volume", variable=self.mode_var, 
                       value="volume").grid(row=0, column=2, padx=20)
        ttk.Radiobutton(config_frame, text="Content of all articles", variable=self.mode_var, 
                       value="all").grid(row=0, column=3, padx=20)
        
        # Lemmatization/Stemming
        ttk.Label(config_frame, text="Lemmatization and Stemming:").grid(row=0, column=4, padx=20)
        self.stemming_var = tk.StringVar(value="None")
        stemming_combo = ttk.Combobox(config_frame, textvariable=self.stemming_var, 
                                      values=["None", "Porter Stemmer"], width=15)
        stemming_combo.current(0)
        stemming_combo.grid(row=0, column=5, padx=5)
        
        # Normalization
        self.norm_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(config_frame, text="Normalization", variable=self.norm_var).grid(row=0, column=6, padx=20)
        
        # ===== Section 2: Visualization =====
        viz_frame = ttk.LabelFrame(main_frame, text="Data Visualization", padding="10")
        viz_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5)
        main_frame.rowconfigure(1, weight=1)
        
        # Bouton Visualization
        ttk.Button(viz_frame, text="Load and Visualize Data", 
                  command=self.load_data, width=30).grid(row=0, column=0, pady=5)
        
        # Table pour afficher les articles
        columns = ("N° Article", "N° label", "Label")
        self.tree = ttk.Treeview(viz_frame, columns=columns, show='headings', height=8)
        self.tree.heading("N° Article", text="N° Article")
        self.tree.heading("N° label", text="N° label")
        self.tree.heading("Label", text="Label")
        
        self.tree.column("N° Article", width=100)
        self.tree.column("N° label", width=100)
        self.tree.column("Label", width=300)
        
        self.tree.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5)
        viz_frame.rowconfigure(1, weight=1)
        viz_frame.columnconfigure(0, weight=1)
        
        # Scrollbar
        scrollbar = ttk.Scrollbar(viz_frame, orient=tk.VERTICAL, command=self.tree.yview)
        scrollbar.grid(row=1, column=1, sticky=(tk.N, tk.S))
        self.tree.configure(yscrollcommand=scrollbar.set)
        
        # ===== Section 3: Training =====
        train_frame = ttk.LabelFrame(main_frame, text="B. Learning (or estimating) the probabilities P(c) and P(w|c)", 
                                     padding="10")
        train_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5)
        main_frame.rowconfigure(2, weight=2)
        
        ttk.Button(train_frame, text="Training", 
                  command=self.train_model, width=30).grid(row=0, column=0, pady=5)
        
        # Frame pour P(c)
        pc_frame = ttk.LabelFrame(train_frame, text="Estimating the class probabilities P(c)", padding="5")
        pc_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=5)
        
        self.pc_text = scrolledtext.ScrolledText(pc_frame, height=3, width=140)
        self.pc_text.pack(fill=tk.BOTH, expand=True)
        
        # Frame pour P(w|c)
        pwc_frame = ttk.LabelFrame(train_frame, text="Estimating the conditional probabilities P(w|c)", padding="5")
        pwc_frame.grid(row=2, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5)
        train_frame.rowconfigure(2, weight=1)
        
        self.pwc_text = scrolledtext.ScrolledText(pwc_frame, height=10, width=140)
        self.pwc_text.pack(fill=tk.BOTH, expand=True)
        
        # ===== Section 4: Testing =====
        test_frame = ttk.LabelFrame(main_frame, text="Testing", padding="10")
        test_frame.grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        
        ttk.Label(test_frame, text="Test article n°:").grid(row=0, column=0, padx=5)
        self.test_article_var = tk.IntVar(value=1)
        ttk.Spinbox(test_frame, from_=1, to=200, textvariable=self.test_article_var, width=10).grid(row=0, column=1, padx=5)
        
        ttk.Button(test_frame, text="Testing", 
                  command=self.test_model, width=30).grid(row=0, column=2, padx=20)
        
        self.result_label = ttk.Label(test_frame, text="", font=('Arial', 11, 'bold'))
        self.result_label.grid(row=0, column=3, padx=20)
        
        # Mapping des classes
        self.class_names = {
            1: "Metaheuristics",
            2: "Machine & Deep Learning",
            3: "Classical Optimization",
            4: "Other"
        }
    
    def load_data(self):
        """Charge les données et affiche les articles"""
        self.tree.delete(*self.tree.get_children())
        
        volume = self.volume_var.get()
        num_articles, num_classes = self.classifier.load_data(volume)
        
        if num_articles == 0:
            messagebox.showwarning("No Data", f"No articles found for volume {volume}")
            return
        
        # Afficher les articles dans le tableau
        for article_num in sorted(self.classifier.articles.keys()):
            label = self.classifier.labels.get(article_num, "N/A")
            label_name = self.class_names.get(label, "Unknown")
            self.tree.insert("", tk.END, values=(article_num, label, label_name))
        
        messagebox.showinfo("Data Loaded", 
                          f"Loaded {num_articles} articles with {num_classes} classes")
    
    def train_model(self):
        """Entraîne le modèle et affiche les probabilités"""
        if not self.classifier.articles:
            messagebox.showwarning("No Data", "Please load data first")
            return
        
        # Configurer les options
        self.classifier.use_stemming = (self.stemming_var.get() == "Porter Stemmer")
        self.classifier.use_normalization = self.norm_var.get()
        
        # Entraîner
        class_probs, word_probs = self.classifier.train()
        
        # Afficher P(c)
        self.pc_text.delete(1.0, tk.END)
        pc_header = f"{'Class':<25} {'1st Class':<15} {'2nd Class':<15} {'3rd Class':<15} {'4th Class':<15}\n"
        pc_header += "-" * 85 + "\n"
        self.pc_text.insert(tk.END, pc_header)
        
        pc_values = f"{'P(c)':<25} "
        for c in sorted(self.classifier.classes):
            pc_values += f"{class_probs[c]:<15.4f} "
        self.pc_text.insert(tk.END, pc_values + "\n")
        
        # Afficher P(w|c) - Top mots
        self.pwc_text.delete(1.0, tk.END)
        pwc_header = f"{'P(w|c)':<20} {'1st Class':<20} {'2nd Class':<20} {'3rd Class':<20} {'4th Class':<20}\n"
        pwc_header += "-" * 100 + "\n"
        self.pwc_text.insert(tk.END, pwc_header)
        
        # Calculer un score pour chaque mot basé sur la variance des probabilités
        word_scores = {}
        for word in word_probs:
            probs = [word_probs[word][c] for c in sorted(self.classifier.classes)]
            # Score basé sur la variance (mots distinctifs)
            mean_prob = sum(probs) / len(probs)
            variance = sum((p - mean_prob) ** 2 for p in probs) / len(probs)
            word_scores[word] = variance
        
        # Trier les mots par score décroissant et prendre le top 50
        top_words = sorted(word_scores.keys(), key=lambda w: word_scores[w], reverse=True)[:50]
        
        # Afficher les mots triés
        for word in top_words:
            line = f"{word:<20} "
            for c in sorted(self.classifier.classes):
                prob = word_probs[word][c]
                line += f"{prob:<20.4f} "
            self.pwc_text.insert(tk.END, line + "\n")
        
        messagebox.showinfo("Training Complete", 
                          f"Model trained with {len(self.classifier.vocabulary)} unique words")
    
    def test_model(self):
        """Teste le modèle sur un article"""
        if not self.classifier.class_probs:
            messagebox.showwarning("No Model", "Please train the model first")
            return
        
        article_num = self.test_article_var.get()
        predicted_class, scores = self.classifier.predict(article_num)
        
        if predicted_class is None:
            messagebox.showwarning("Article Not Found", 
                                 f"Article {article_num} not found in the dataset")
            return
        
        actual_class = self.classifier.labels.get(article_num, "N/A")
        predicted_name = self.class_names.get(predicted_class, "Unknown")
        actual_name = self.class_names.get(actual_class, "Unknown")
        
        # Afficher le résultat
        if predicted_class == actual_class:
            result_text = f"✓ Correct! Predicted: {predicted_name} (Class {predicted_class})"
            self.result_label.config(text=result_text, foreground="green")
        else:
            result_text = f"✗ Wrong! Predicted: {predicted_name} (Class {predicted_class}), Actual: {actual_name} (Class {actual_class})"
            self.result_label.config(text=result_text, foreground="red")
        
        # Afficher les scores dans une popup
        scores_text = "Classification Scores:\n\n"
        for c in sorted(scores.keys()):
            scores_text += f"Class {c} ({self.class_names.get(c, 'Unknown')}): {scores[c]:.4f}\n"
        
        messagebox.showinfo("Prediction Details", scores_text)


def main():
    root = tk.Tk()
    app = NaiveBayesGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
