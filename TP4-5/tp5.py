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
        
        # Dossiers pour les données de test (dans TP5)
        base_dir = os.path.dirname(script_dir)
        self.test_articles_dir = os.path.join(base_dir, "TP5", "All-in-many_classification_testing")
        self.test_labels_dir = os.path.join(base_dir, "TP5", "All-in-many_classification_testing")
        
        # Données d'entraînement
        self.train_articles = {}
        self.train_labels = {}
        
        # Données de test
        self.test_articles = {}
        self.test_labels = {}
        
        self.classes = set()
        
        # Probabilités (apprises sur les données d'entraînement)
        self.class_probs = {}
        self.word_given_class_probs = defaultdict(lambda: defaultdict(float))
        self.vocabulary = set()
        
        # Options
        self.use_stemming = False
        self.use_normalization = False
        self.stemmer = PorterStemmer()
        
        # Tokenizer avec expression régulière personnalisée
        self.tokenizer = nltk.RegexpTokenizer(r"(?:[A-Za-z]\.)+|[A-Za-z]+[-@]\d+(?:\.\d+)?|\d+[A-Za-z]+|\d+(?:[.,-]\d+)?%?|\w+(?:[-/]\w+)*|[.!?]+")
        
        # Stop words
        self.stop_words = set(stopwords.words('english'))
        
    def load_training_data(self, volume=18):
        """Charge les données d'entraînement (Volume 18)"""
        self.train_articles.clear()
        self.train_labels.clear()
        self.classes.clear()
        
        pattern = re.compile(rf"Article_(\d+)_Volume_{volume}$")
        
        # Charger les articles d'entraînement
        if os.path.exists(self.articles_dir):
            for filename in os.listdir(self.articles_dir):
                match = pattern.match(filename)
                if match:
                    article_num = int(match.group(1))
                    filepath = os.path.join(self.articles_dir, filename)
                    with open(filepath, 'r', encoding='utf-8') as f:
                        self.train_articles[article_num] = f.read().strip()
        
        # Charger les labels d'entraînement
        if os.path.exists(self.labels_dir):
            for filename in os.listdir(self.labels_dir):
                match = pattern.match(filename)
                if match:
                    article_num = int(match.group(1))
                    filepath = os.path.join(self.labels_dir, filename)
                    with open(filepath, 'r', encoding='utf-8') as f:
                        label = int(f.read().strip())
                        self.train_labels[article_num] = label
                        self.classes.add(label)
        
        return len(self.train_articles), len(self.classes)
    
    def load_testing_data(self, volumes=[17, 14]):
        """Charge les données de test depuis TP5"""
        self.test_articles.clear()
        self.test_labels.clear()
        
        # Pattern pour les fichiers: Article_1, Article_2, etc.
        article_pattern = re.compile(r"^Article_(\d+)$")
        label_pattern = re.compile(r"^Article_(\d+)_label$")
        
        if not os.path.exists(self.test_articles_dir):
            return 0
        
        # Charger tous les fichiers
        for filename in os.listdir(self.test_articles_dir):
            # Vérifier si c'est un article (sans _label)
            article_match = article_pattern.match(filename)
            if article_match:
                article_num = int(article_match.group(1))
                key = f"test_{article_num}"
                
                # Lire le contenu de l'article
                filepath = os.path.join(self.test_articles_dir, filename)
                with open(filepath, 'r', encoding='utf-8') as f:
                    self.test_articles[key] = f.read().strip()
                
                # Lire le label correspondant
                label_filename = f"Article_{article_num}_label"
                label_filepath = os.path.join(self.test_labels_dir, label_filename)
                if os.path.exists(label_filepath):
                    with open(label_filepath, 'r', encoding='utf-8') as f:
                        label = int(f.read().strip())
                        self.test_labels[key] = label
        
        return len(self.test_articles)
    
    def preprocess_text(self, text):
        """Prétraite le texte : tokenisation, normalisation, stemming"""
        # Utiliser le tokenizer NLTK avec l'expression régulière personnalisée
        words = self.tokenizer.tokenize(text.lower())
        
        if self.use_stemming:
            words = [self.stemmer.stem(word) for word in words]
        
        return words
    
    def train(self):
        """Entraîne le classificateur Naive Bayes sur les données d'entraînement"""
        # Compter les documents par classe
        class_counts = Counter(self.train_labels.values())
        total_docs = len(self.train_labels)
        
        # Calculer P(c)
        self.class_probs = {c: count / total_docs for c, count in class_counts.items()}
        
        # Construire le vocabulaire et compter les mots par classe
        word_counts_by_class = defaultdict(Counter)
        total_words_by_class = defaultdict(int)
        
        for article_num, label in self.train_labels.items():
            if article_num in self.train_articles:
                words = self.preprocess_text(self.train_articles[article_num])
                self.vocabulary.update(words)
                word_counts_by_class[label].update(words)
                total_words_by_class[label] += len(words)
        
        vocab_size = len(self.vocabulary)
        
        # Calculer P(w|c) avec lissage de Laplace
        for c in self.classes:
            for word in self.vocabulary:
                count = word_counts_by_class[c][word]
                # Lissage de Laplace : (count + 1) / (total + vocab_size)
                self.word_given_class_probs[word][c] = (count + 1) / (total_words_by_class[c] + vocab_size)
        
        return self.class_probs, dict(self.word_given_class_probs)
    
    def predict(self, text):
        """Prédit la classe d'un texte donné"""
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
    
    def evaluate_on_test_set(self):
        """Évalue le modèle sur l'ensemble de test et retourne les résultats détaillés"""
        if not self.class_probs:
            return None
        
        results = []
        correct = 0
        total = 0
        
        for key, text in self.test_articles.items():
            if key in self.test_labels:
                actual_class = self.test_labels[key]
                predicted_class, scores = self.predict(text)
                
                is_correct = (predicted_class == actual_class)
                if is_correct:
                    correct += 1
                total += 1
                
                results.append({
                    'key': key,
                    'text_preview': text[:100] + "..." if len(text) > 100 else text,
                    'actual': actual_class,
                    'predicted': predicted_class,
                    'correct': is_correct,
                    'scores': scores
                })
        
        accuracy = (correct / total * 100) if total > 0 else 0
        
        return {
            'results': results,
            'correct': correct,
            'total': total,
            'accuracy': accuracy
        }


class TestingGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Part 5 - Text Classification - Testing (Naive Bayes)")
        self.root.geometry("1400x900")
        
        self.classifier = NaiveBayesClassifier()
        
        # Mapping des classes
        self.class_names = {
            1: "Metaheuristics",
            2: "Machine & Deep Learning",
            3: "Classical Optimization",
            4: "Other"
        }
        
        self.create_widgets()
        
    def create_widgets(self):
        # Frame principal
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configuration de la grille
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        
        # ===== Section 1: Configuration =====
        config_frame = ttk.LabelFrame(main_frame, text="Configuration", padding="10")
        config_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=5)
        
        # Training Volume
        ttk.Label(config_frame, text="Training Volume:").grid(row=0, column=0, padx=5)
        self.train_volume_var = tk.IntVar(value=18)
        ttk.Label(config_frame, text="18", font=('Arial', 10, 'bold')).grid(row=0, column=1, padx=5)
        
        # Testing Volumes
        ttk.Label(config_frame, text="Testing Volumes:").grid(row=0, column=2, padx=20)
        ttk.Label(config_frame, text="17, 14", font=('Arial', 10, 'bold')).grid(row=0, column=3, padx=5)
        
        # Lemmatization/Stemming
        ttk.Label(config_frame, text="Lemmatization and Stemming:").grid(row=0, column=4, padx=20)
        self.stemming_var = tk.StringVar(value="porter")
        stemming_combo = ttk.Combobox(config_frame, textvariable=self.stemming_var, 
                                      values=["None", "Porter Stemmer"], width=15)
        stemming_combo.current(1)
        stemming_combo.grid(row=0, column=5, padx=5)
        
        # Normalization
        self.norm_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(config_frame, text="Normalization", variable=self.norm_var).grid(row=0, column=6, padx=20)
        
        # ===== Section 2: Training =====
        train_frame = ttk.LabelFrame(main_frame, text="1. Training Phase (Volume 18)", padding="10")
        train_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=5)
        
        ttk.Button(train_frame, text="Load Training Data and Train Model", 
                  command=self.train_model, width=40).grid(row=0, column=0, pady=5)
        
        self.train_info = ttk.Label(train_frame, text="", font=('Arial', 10))
        self.train_info.grid(row=0, column=1, padx=20)
        
        # ===== Section 3: Testing =====
        test_frame = ttk.LabelFrame(main_frame, text="2. Testing Phase (Volumes 17 & 14)", padding="10")
        test_frame.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=5)
        
        ttk.Button(test_frame, text="Load Test Data and Evaluate", 
                  command=self.test_model, width=40).grid(row=0, column=0, pady=5)
        
        self.test_info = ttk.Label(test_frame, text="", font=('Arial', 10))
        self.test_info.grid(row=0, column=1, padx=20)
        
        # ===== Section 4: Results =====
        results_frame = ttk.LabelFrame(main_frame, text="3. Test Results", padding="10")
        results_frame.grid(row=3, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5)
        main_frame.rowconfigure(3, weight=1)
        
        # Accuracy Label
        self.accuracy_label = ttk.Label(results_frame, text="", 
                                       font=('Arial', 14, 'bold'), foreground="blue")
        self.accuracy_label.pack(pady=10)
        
        # Table pour afficher les résultats
        columns = ("Article", "Actual Class", "Predicted Class", "Status", "Preview")
        self.results_tree = ttk.Treeview(results_frame, columns=columns, show='headings', height=20)
        
        self.results_tree.heading("Article", text="Article")
        self.results_tree.heading("Actual Class", text="Actual Class")
        self.results_tree.heading("Predicted Class", text="Predicted Class")
        self.results_tree.heading("Status", text="Status")
        self.results_tree.heading("Preview", text="Text Preview")
        
        self.results_tree.column("Article", width=120)
        self.results_tree.column("Actual Class", width=200)
        self.results_tree.column("Predicted Class", width=200)
        self.results_tree.column("Status", width=100)
        self.results_tree.column("Preview", width=400)
        
        self.results_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Scrollbar
        scrollbar = ttk.Scrollbar(results_frame, orient=tk.VERTICAL, command=self.results_tree.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.results_tree.configure(yscrollcommand=scrollbar.set)
        
        # Tags pour les couleurs
        self.results_tree.tag_configure('correct', background='#90EE90')  # Vert clair
        self.results_tree.tag_configure('incorrect', background='#FFB6C1')  # Rose clair
        
        # ===== Section 5: Single Article Test =====
        single_test_frame = ttk.LabelFrame(main_frame, text="4. Test Single Article", padding="10")
        single_test_frame.grid(row=4, column=0, sticky=(tk.W, tk.E), pady=5)
        
        ttk.Label(single_test_frame, text="Test text:").grid(row=0, column=0, padx=5)
        
        self.single_test_text = scrolledtext.ScrolledText(single_test_frame, height=3, width=100)
        self.single_test_text.grid(row=1, column=0, columnspan=3, padx=5, pady=5)
        
        ttk.Button(single_test_frame, text="Classify Text", 
                  command=self.classify_single_text, width=30).grid(row=2, column=0, pady=5)
        
        self.single_result_label = ttk.Label(single_test_frame, text="", font=('Arial', 11, 'bold'))
        self.single_result_label.grid(row=2, column=1, padx=20)
    
    def train_model(self):
        """Entraîne le modèle sur les données d'entraînement"""
        # Configurer les options
        self.classifier.use_stemming = (self.stemming_var.get() == "Porter Stemmer")
        self.classifier.use_normalization = self.norm_var.get()
        
        # Charger les données d'entraînement
        num_articles, num_classes = self.classifier.load_training_data(volume=18)
        
        if num_articles == 0:
            messagebox.showwarning("No Data", "No training articles found")
            return
        
        # Entraîner le modèle
        class_probs, word_probs = self.classifier.train()
        
        info_text = f"✓ Training complete: {num_articles} articles, {num_classes} classes, {len(self.classifier.vocabulary)} unique words"
        self.train_info.config(text=info_text, foreground="green")
        
        messagebox.showinfo("Training Complete", 
                          f"Model trained successfully!\n\n"
                          f"Articles: {num_articles}\n"
                          f"Classes: {num_classes}\n"
                          f"Vocabulary size: {len(self.classifier.vocabulary)}")
    
    def test_model(self):
        """Teste le modèle sur les données de test"""
        if not self.classifier.class_probs:
            messagebox.showwarning("No Model", "Please train the model first")
            return
        
        # Charger les données de test
        num_test = self.classifier.load_testing_data(volumes=[17, 14])
        
        if num_test == 0:
            messagebox.showwarning("No Data", "No test articles found")
            return
        
        info_text = f"✓ Test data loaded: {num_test} articles from volumes 17 & 14"
        self.test_info.config(text=info_text, foreground="green")
        
        # Évaluer sur l'ensemble de test
        evaluation = self.classifier.evaluate_on_test_set()
        
        if evaluation is None:
            messagebox.showerror("Error", "Evaluation failed")
            return
        
        # Effacer les résultats précédents
        self.results_tree.delete(*self.results_tree.get_children())
        
        # Afficher l'accuracy
        accuracy_text = f"Accuracy: {evaluation['accuracy']:.2f}% ({evaluation['correct']}/{evaluation['total']} correct)"
        self.accuracy_label.config(text=accuracy_text)
        
        # Afficher les résultats détaillés
        for result in evaluation['results']:
            article_key = result['key']
            actual_name = self.class_names.get(result['actual'], "Unknown")
            predicted_name = self.class_names.get(result['predicted'], "Unknown")
            status = "✓ Correct" if result['correct'] else "✗ Wrong"
            preview = result['text_preview']
            
            tag = 'correct' if result['correct'] else 'incorrect'
            
            self.results_tree.insert("", tk.END, 
                                    values=(article_key, 
                                           f"Class {result['actual']}: {actual_name}",
                                           f"Class {result['predicted']}: {predicted_name}",
                                           status,
                                           preview),
                                    tags=(tag,))
        
        messagebox.showinfo("Testing Complete", 
                          f"Evaluation completed!\n\n"
                          f"Total articles tested: {evaluation['total']}\n"
                          f"Correct predictions: {evaluation['correct']}\n"
                          f"Wrong predictions: {evaluation['total'] - evaluation['correct']}\n"
                          f"Accuracy: {evaluation['accuracy']:.2f}%")
    
    def classify_single_text(self):
        """Classifie un texte saisi par l'utilisateur"""
        if not self.classifier.class_probs:
            messagebox.showwarning("No Model", "Please train the model first")
            return
        
        text = self.single_test_text.get(1.0, tk.END).strip()
        
        if not text:
            messagebox.showwarning("No Text", "Please enter text to classify")
            return
        
        predicted_class, scores = self.classifier.predict(text)
        predicted_name = self.class_names.get(predicted_class, "Unknown")
        
        result_text = f"Classification result: class n°{predicted_class}, {predicted_name}"
        self.single_result_label.config(text=result_text, foreground="blue")
        
        # Afficher les scores détaillés
        scores_text = "Classification Scores:\n\n"
        for c in sorted(scores.keys()):
            scores_text += f"Class {c} ({self.class_names.get(c, 'Unknown')}): {scores[c]:.4f}\n"
        
        messagebox.showinfo("Prediction Details", scores_text)


def main():
    root = tk.Tk()
    app = TestingGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
