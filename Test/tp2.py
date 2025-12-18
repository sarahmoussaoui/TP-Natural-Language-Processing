import os
import time
import threading
import requests
from urllib.parse import urljoin
from bs4 import BeautifulSoup
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from collections import Counter
from nltk.tokenize import RegexpTokenizer
from nltk.stem import PorterStemmer, LancasterStemmer, SnowballStemmer
from nltk.util import ngrams
import re

# ----------------- Configuration -----------------
BASE_ISSUES_URL = "https://link.springer.com/journal/12065/volumes-and-issues"
HEADERS = {"User-Agent": "Mozilla/5.0"}
DELAY = 1.0
OUTPUT_DIR = os.path.join(os.path.expanduser("~"), "TalnAppOutput")
os.makedirs(OUTPUT_DIR, exist_ok=True)

TOKEN_PATTERN = r"(?:[A-Za-z]\.)+|[A-Za-z]+[\-@]\d+(?:\.\d+)?|\d+[A-Za-z]+|\d+(?:[\.\,\-]\d+)?%?|\w+(?:[\-/]\w+)*"
tokenizer = RegexpTokenizer(TOKEN_PATTERN)

STEMMERS = {
    "Aucun": None,
    "Porter": PorterStemmer(),
    "Lancaster": LancasterStemmer(),
    "Snowball": SnowballStemmer("english")
}

# ----------------- Fonctions d'extraction -----------------
def fetch_soup(url):
    r = requests.get(url, headers=HEADERS, timeout=20)
    r.raise_for_status()
    return BeautifulSoup(r.text, "html.parser")

def list_volumes_and_issues():
    soup = fetch_soup(BASE_ISSUES_URL)
    volumes = []
    for vol_section in soup.find_all(["h2"]):
        vol_text = vol_section.get_text(strip=True)
        if vol_text.lower().startswith("volume"):
            vol_match = re.search(r'volume\s*(\d+)', vol_text, re.IGNORECASE)
            if vol_match:
                vol_num = vol_match.group(1)
                clean_vol_text = f"Volume {vol_num}"
            else:
                num_match = re.search(r'(\d+)', vol_text)
                if num_match:
                    clean_vol_text = f"Volume {num_match.group(1)}"
                else:
                    clean_vol_text = vol_text.split()[0]
            
            issues = []
            el = vol_section.find_next_sibling()
            issue_count = 0
            while el and el.name != "h2":
                for a in el.find_all("a", href=True):
                    href = a["href"]
                    if "/volumes-and-issues/" in href or "/issue/" in href or "/issue" in href:
                        full = urljoin(BASE_ISSUES_URL, href.split("?")[0])
                        issue_name = a.get_text(strip=True)
                        
                        issue_match = re.search(r'issue\s*(\d+(?:[\-\–]\d+)?)', issue_name, re.IGNORECASE)
                        if issue_match:
                            issue_num = issue_match.group(1)
                            clean_issue = f"Issue {issue_num}"
                        else:
                            issue_count += 1
                            clean_issue = f"Issue {issue_count}"
                        
                        issues.append((clean_issue, full))
                el = el.find_next_sibling()
            if issues:
                volumes.append((clean_vol_text, issues))
    
    if not volumes:
        for line in soup.get_text("\n").splitlines():
            if line.strip().lower().startswith("volume"):
                vol_match = re.search(r'volume\s*(\d+)', line.strip(), re.IGNORECASE)
                if vol_match:
                    clean_line = f"Volume {vol_match.group(1)}"
                else:
                    clean_line = line.strip().split()[0]
                volumes.append((clean_line, []))
    
    return volumes

def list_articles_in_issue(issue_url):
    soup = fetch_soup(issue_url)
    links = []
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if "/article/" in href:
            full = urljoin(issue_url, href.split("?")[0])
            if full not in links:
                links.append(full)
    articles = []
    for link in links:
        try:
            page = fetch_soup(link)
            t = page.find("h1")
            title = t.get_text(strip=True) if t else link
            articles.append((title, link))
            time.sleep(DELAY)
        except Exception:
            articles.append((link, link))
    return articles

def extract_title_and_abstract(article_url):
    soup = fetch_soup(article_url)
    t = soup.find("h1")
    title = t.get_text(strip=True) if t else (soup.title.string.strip() if soup.title else "(sans titre)")
    abstract = ""
    for tag in soup.find_all(["div", "section"]):
        cls = tag.get("class") or []
        clsj = " ".join(cls).lower()
        if "c-article-section" in clsj:
            txt = tag.get_text(" ", strip=True)
            if txt and len(txt) > 50:
                abstract = txt
                break
    if not abstract:
        meta = soup.find("meta", property="og:description") or soup.find("meta", attrs={"name":"dc.Description"}) or soup.find("meta", attrs={"name":"description"})
        if meta and meta.get("content"):
            abstract = meta["content"].strip()
    if not abstract:
        main = soup.find("main") or soup.find("article")
        if main:
            abstract = main.get_text(" ", strip=True)[:1200]
            
    if abstract:
        abstract = re.sub(r'^abstract\s*[:\-]?\s*', '', abstract, flags=re.IGNORECASE)
        abstract = abstract.strip()
        
    return title, abstract

# ----------------- Traitement texte & n-grams -----------------
def tokenize_and_normalize(text, stemmer_name):
    toks = tokenizer.tokenize(text)
    toks = [t for t in toks if t.strip()]
    if stemmer_name == "Aucun":
        return toks
    stemmer = STEMMERS.get(stemmer_name)
    if stemmer is None:
        return toks
    return [stemmer.stem(t.lower()) for t in toks]

def compute_ngrams(tokens, n):
    if not tokens or len(tokens) < n:
        return {}
    grams = list(ngrams(tokens, n))
    counter = Counter(grams)
    if n == 1:
        total = len(tokens)
        return {gram[0] if isinstance(gram, tuple) else gram: (freq, freq/total) for gram, freq in counter.items()}
    elif n == 2:
        unigrams = Counter(tokens)
        result = {}
        for (w1, w2), freq in counter.items():
            prob = freq / unigrams[w1] if unigrams[w1] > 0 else 0
            result[(w1, w2)] = (freq, prob)
        return result
    elif n == 3:
        bigrams = Counter(ngrams(tokens, 2))
        result = {}
        for (w1, w2, w3), freq in counter.items():
            bigram_count = bigrams.get((w1, w2), 0)
            prob = freq / bigram_count if bigram_count > 0 else 0
            result[(w1, w2, w3)] = (freq, prob)
        return result
    return {}

# ----------------- Interface Graphique Redesignée -----------------
class ExtracteurNgrams(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Analyseur N-grams | Publications Scientifiques Springer")
        self.geometry("1300x850")
        self.minsize(1100, 750)
        self._configurer_style()
        self._creer_interface()
        self.volumes = []
        threading.Thread(target=self.charger_volumes, daemon=True).start()

    def _configurer_style(self):
        style = ttk.Style(self)
        style.theme_use("clam")

        # Palette de couleurs moderne (bleu/vert/gris)
        BLEU_PRINCIPAL = "#2563eb"
        BLEU_FONCE = "#1e40af"
        VERT_ACCENT = "#10b981"
        GRIS_CLAIR = "#f8fafc"
        GRIS_MOYEN = "#e2e8f0"
        TEXTE_FONCE = "#1e293b"
        TEXTE_CLAIR = "#64748b"

        style.configure("TFrame", background=GRIS_CLAIR)
        style.configure("TLabel", background=GRIS_CLAIR, foreground=TEXTE_FONCE, 
                       font=("Helvetica", 10))
        
        # Card style
        style.configure("Card.TFrame", background="white", relief="flat", borderwidth=1)
        
        # Titres
        style.configure("Titre.TLabel", font=("Helvetica", 16, "bold"),
                       background="white", foreground=BLEU_PRINCIPAL, padding=10)
        style.configure("SousTitre.TLabel", font=("Helvetica", 11, "bold"),
                       background="white", foreground=TEXTE_FONCE, padding=6)
        style.configure("Section.TLabel", font=("Helvetica", 9, "bold"),
                       background="white", foreground=TEXTE_CLAIR)
        
        # Boutons
        style.configure("Action.TButton", font=("Helvetica", 10, "bold"),
                       background=BLEU_PRINCIPAL, foreground="white",
                       borderwidth=0, padding=8)
        style.map("Action.TButton",
                 background=[("active", BLEU_FONCE)],
                 relief=[("pressed", "flat")])
        
        style.configure("Secondary.TButton", font=("Helvetica", 9),
                       background=GRIS_MOYEN, foreground=TEXTE_FONCE,
                       borderwidth=0, padding=6)
        style.map("Secondary.TButton",
                 background=[("active", "#cbd5e1")])
        
        style.configure("Success.TButton", font=("Helvetica", 10, "bold"),
                       background=VERT_ACCENT, foreground="white",
                       borderwidth=0, padding=8)
        style.map("Success.TButton",
                 background=[("active", "#059669")])
        
        # Combobox
        style.configure("TCombobox", font=("Helvetica", 9),
                       fieldbackground="white", background="white")
        
        # Treeview
        style.configure("Treeview", font=("Helvetica", 9), rowheight=26,
                       background="white", fieldbackground="white",
                       foreground=TEXTE_FONCE, borderwidth=0)
        style.configure("Treeview.Heading", font=("Helvetica", 10, "bold"),
                       background=GRIS_MOYEN, foreground=TEXTE_FONCE,
                       relief="flat")
        style.map("Treeview", background=[("selected", BLEU_PRINCIPAL)],
                 foreground=[("selected", "white")])
        
        # Notebook
        style.configure("TNotebook", background="white", borderwidth=0, padding=0)
        style.configure("TNotebook.Tab", font=("Helvetica", 10, "bold"),
                       padding=[14, 6], background=GRIS_MOYEN, 
                       foreground=TEXTE_CLAIR)
        style.map("TNotebook.Tab",
                 background=[("selected", "white")],
                 foreground=[("selected", BLEU_PRINCIPAL)])
        
        # Statut
        style.configure("Statut.TLabel", font=("Helvetica", 9),
                       background=GRIS_MOYEN, foreground=TEXTE_CLAIR,
                       padding=6)

    def _creer_interface(self):
        # Conteneur principal avec padding
        principal = ttk.Frame(self, style="TFrame")
        principal.pack(fill="both", expand=True, padx=12, pady=12)

        # === EN-TÊTE ===
        carte_entete = ttk.Frame(principal, style="Card.TFrame", relief="solid", borderwidth=1)
        carte_entete.pack(fill="x", pady=(0, 10))
        
        ttk.Label(carte_entete, text="📚 Analyseur de N-grams", 
                 style="Titre.TLabel").pack(anchor="w")
        ttk.Label(carte_entete, text="Extraction et analyse linguistique depuis Springer Journal",
                 style="Section.TLabel").pack(anchor="w", padx=10, pady=(0, 6))

        # === LAYOUT HORIZONTAL : GAUCHE (Config + Articles) | DROITE (Résultats) ===
        conteneur_split = ttk.Frame(principal, style="TFrame")
        conteneur_split.pack(fill="both", expand=True)

        # ========== PARTIE GAUCHE ==========
        gauche = ttk.Frame(conteneur_split, style="TFrame")
        gauche.pack(side="left", fill="both", expand=False, padx=(0, 6))
        gauche.configure(width=450)

        # --- Section Configuration ---
        carte_config = ttk.Frame(gauche, style="Card.TFrame", relief="solid", borderwidth=1)
        carte_config.pack(fill="x", pady=(0, 10))
        
        ttk.Label(carte_config, text="⚙️ Configuration",
                 style="SousTitre.TLabel").pack(anchor="w")
        
        grille_config = ttk.Frame(carte_config, style="Card.TFrame")
        grille_config.pack(fill="x", padx=10, pady=(0, 10))
        
        ttk.Label(grille_config, text="Volume/Numéro :", 
                 background="white", font=("Helvetica", 9)).grid(row=0, column=0, sticky="w", pady=6)
        self.combo_issue = ttk.Combobox(grille_config, values=[], width=35, state="readonly")
        self.combo_issue.grid(row=0, column=1, sticky="ew", padx=(8, 0), pady=6)
        
        ttk.Label(grille_config, text="Stemming :", 
                 background="white", font=("Helvetica", 9)).grid(row=1, column=0, sticky="w", pady=6)
        self.var_stemmer = tk.StringVar(value="Aucun")
        combo_stem = ttk.Combobox(grille_config, textvariable=self.var_stemmer, 
                                  values=list(STEMMERS.keys()), state="readonly", width=20)
        combo_stem.grid(row=1, column=1, sticky="w", padx=(8, 0), pady=6)
        
        grille_config.columnconfigure(1, weight=1)
        
        ttk.Button(carte_config, text="→ Charger les articles", 
                  command=self.action_charger_articles, 
                  style="Action.TButton").pack(fill="x", padx=10, pady=(0, 10))

        # --- Section Liste des Articles ---
        carte_articles = ttk.Frame(gauche, style="Card.TFrame", relief="solid", borderwidth=1)
        carte_articles.pack(fill="both", expand=True)
        
        entete_articles = ttk.Frame(carte_articles, style="Card.TFrame")
        entete_articles.pack(fill="x")
        
        ttk.Label(entete_articles, text="📋 Articles disponibles",
                 style="SousTitre.TLabel").pack(side="left")
        
        boutons_articles = ttk.Frame(entete_articles, style="Card.TFrame")
        boutons_articles.pack(side="right", padx=10, pady=6)
        
        ttk.Button(boutons_articles, text="Analyser", 
                  command=self.action_generer_ngrams,
                  style="Success.TButton").pack(side="left", padx=2)
        ttk.Button(boutons_articles, text="Exporter tout", 
                  command=self.action_exporter_tout,
                  style="Secondary.TButton").pack(side="left", padx=2)
        
        cadre_liste = ttk.Frame(carte_articles, style="Card.TFrame")
        cadre_liste.pack(fill="both", expand=True, padx=10, pady=(0, 10))
        
        scroll_articles = ttk.Scrollbar(cadre_liste)
        scroll_articles.pack(side="right", fill="y")
        
        self.liste_articles = tk.Listbox(cadre_liste, font=("Helvetica", 9),
                                        yscrollcommand=scroll_articles.set,
                                        selectmode="browse", activestyle="none",
                                        relief="solid", borderwidth=1,
                                        highlightthickness=0)
        self.liste_articles.pack(side="left", fill="both", expand=True)
        scroll_articles.config(command=self.liste_articles.yview)

        # ========== PARTIE DROITE (Résultats) ==========
        droite = ttk.Frame(conteneur_split, style="TFrame")
        droite.pack(side="right", fill="both", expand=True)

        carte_resultats = ttk.Frame(droite, style="Card.TFrame", relief="solid", borderwidth=1)
        carte_resultats.pack(fill="both", expand=True)
        
        ttk.Label(carte_resultats, text="📊 Résultats de l'analyse",
                 style="SousTitre.TLabel").pack(anchor="w")
        
        # Titre de l'article
        self.label_titre = ttk.Label(carte_resultats, text="→ Sélectionnez un article pour commencer",
                                     background="white", foreground="#64748b",
                                     font=("Helvetica", 10, "italic"), wraplength=800)
        self.label_titre.pack(anchor="w", padx=10, pady=(6, 2))
        
        # Résumé (plus compact)
        cadre_resume = ttk.Frame(carte_resultats, style="Card.TFrame")
        cadre_resume.pack(fill="x", padx=10, pady=(4, 8))
        
        ttk.Label(cadre_resume, text="Résumé :", background="white",
                 font=("Helvetica", 9, "bold")).pack(anchor="w", pady=(0, 3))
        
        scroll_resume = ttk.Scrollbar(cadre_resume, orient="vertical")
        self.texte_resume = tk.Text(cadre_resume, height=4, wrap="word",
                                    font=("Helvetica", 9),
                                    yscrollcommand=scroll_resume.set,
                                    relief="solid", borderwidth=1)
        scroll_resume.config(command=self.texte_resume.yview)
        self.texte_resume.pack(side="left", fill="both", expand=True)
        scroll_resume.pack(side="right", fill="y")
        
        # Onglets pour n-grams (PREND TOUT L'ESPACE RESTANT)
        self.onglets = ttk.Notebook(carte_resultats)
        self.onglets.pack(fill="both", expand=True, padx=10, pady=(0, 10))
        
        self.onglet_uni = ttk.Frame(self.onglets, style="Card.TFrame")
        self.onglet_bi = ttk.Frame(self.onglets, style="Card.TFrame")
        self.onglet_tri = ttk.Frame(self.onglets, style="Card.TFrame")
        
        self.onglets.add(self.onglet_uni, text="  Unigrammes  ")
        self.onglets.add(self.onglet_bi, text="  Bigrammes  ")
        self.onglets.add(self.onglet_tri, text="  Trigrammes  ")
        
        # Créer les tableaux
        self.tableaux = {}
        for onglet, nom in ((self.onglet_uni, "uni"), (self.onglet_bi, "bi"), 
                            (self.onglet_tri, "tri")):
            conteneur = ttk.Frame(onglet, style="Card.TFrame")
            conteneur.pack(fill="both", expand=True, padx=6, pady=6)
            
            cols = ("ngram", "freq", "prob")
            arbre = ttk.Treeview(conteneur, columns=cols, show="headings", 
                                selectmode="browse")
            arbre.heading("ngram", text="N-gramme")
            arbre.heading("freq", text="Fréquence")
            arbre.heading("prob", text="Probabilité")
            arbre.column("ngram", width=500, anchor="w")
            arbre.column("freq", width=100, anchor="center")
            arbre.column("prob", width=100, anchor="center")
            
            # Couleurs alternées
            arbre.tag_configure("pair", background="#f8fafc")
            arbre.tag_configure("impair", background="white")
            
            scroll_v = ttk.Scrollbar(conteneur, orient="vertical", command=arbre.yview)
            scroll_h = ttk.Scrollbar(conteneur, orient="horizontal", command=arbre.xview)
            arbre.configure(yscrollcommand=scroll_v.set, xscrollcommand=scroll_h.set)
            
            arbre.grid(row=0, column=0, sticky="nsew")
            scroll_v.grid(row=0, column=1, sticky="ns")
            scroll_h.grid(row=1, column=0, sticky="ew")
            
            conteneur.rowconfigure(0, weight=1)
            conteneur.columnconfigure(0, weight=1)
            
            self.tableaux[nom] = arbre

        # === BARRE DE STATUT ===
        self.var_statut = tk.StringVar(value="⚡ Prêt à analyser")
        barre_statut = ttk.Label(principal, textvariable=self.var_statut,
                                style="Statut.TLabel", relief="flat")
        barre_statut.pack(fill="x", pady=(6, 0))

    # ----------------- Actions -----------------
    def charger_volumes(self):
        try:
            self.var_statut.set("⏳ Chargement des volumes en cours...")
            self.volumes = list_volumes_and_issues()
            liste_issues = []
            self.map_issues = {}
            for titre_vol, issues in self.volumes:
                for nom_issue, url in issues:
                    affichage = f"{titre_vol} → {nom_issue}"
                    liste_issues.append(affichage)
                    self.map_issues[affichage] = url
            self.combo_issue["values"] = liste_issues
            if liste_issues:
                self.combo_issue.set(liste_issues[0])
            self.var_statut.set(f"✓ {len(liste_issues)} numéros disponibles")
        except Exception as e:
            self.var_statut.set("✗ Erreur lors du chargement")
            messagebox.showerror("Erreur", f"Impossible de charger les volumes:\n{e}")

    def action_charger_articles(self):
        selection = self.combo_issue.get()
        if not selection:
            messagebox.showwarning("Attention", "Veuillez sélectionner un volume/numéro.")
            return
        url = self.map_issues.get(selection)
        if not url:
            messagebox.showerror("Erreur", "URL introuvable.")
            return
        threading.Thread(target=self._thread_charger_articles, args=(url,), daemon=True).start()

    def _thread_charger_articles(self, url):
        try:
            self.var_statut.set("⏳ Récupération des articles...")
            articles = list_articles_in_issue(url)
            self.articles = articles
            self.liste_articles.delete(0, tk.END)
            for titre, lien in articles:
                self.liste_articles.insert(tk.END, f"  {titre}")
            self.var_statut.set(f"✓ {len(articles)} articles chargés")
        except Exception as e:
            self.var_statut.set("✗ Erreur chargement articles")
            messagebox.showerror("Erreur", f"Impossible de charger les articles:\n{e}")

    def action_generer_ngrams(self):
        selection_idx = self.liste_articles.curselection()
        if not selection_idx:
            messagebox.showwarning("Attention", 
                                 "Veuillez sélectionner un article dans la liste.")
            return
        idx = selection_idx[0]
        titre, lien = self.articles[idx]
        threading.Thread(target=self._thread_traiter_article, 
                        args=(titre, lien), daemon=True).start()

    def _thread_traiter_article(self, titre, lien):
        try:
            self.var_statut.set("⏳ Analyse en cours...")
            self.label_titre.config(text=f"📄 {titre}", foreground="#1e293b", 
                                   font=("Helvetica", 10, "bold"))
            t, resume = extract_title_and_abstract(lien)
            self.texte_resume.delete("1.0", tk.END)
            self.texte_resume.insert(tk.END, resume[:4000])
            
            choix_stem = self.var_stemmer.get()
            tokens = tokenize_and_normalize(resume, choix_stem)
            uni = compute_ngrams(tokens, 1)
            bi = compute_ngrams(tokens, 2)
            tri = compute_ngrams(tokens, 3)
            
            self._remplir_tableau("uni", uni)
            self._remplir_tableau("bi", bi)
            self._remplir_tableau("tri", tri)
            
            self.var_statut.set("✓ Analyse terminée avec succès")
        except Exception as e:
            self.var_statut.set("✗ Erreur pendant l'analyse")
            messagebox.showerror("Erreur", f"Erreur pendant le traitement:\n{e}")

    def _remplir_tableau(self, nom, dict_ngrams):
        arbre = self.tableaux.get(nom)
        if arbre is None:
            return
        arbre.delete(*arbre.get_children())
        items_tries = sorted(dict_ngrams.items(), key=lambda x: x[1][0], reverse=True)
        for idx, (gram, (freq, prob)) in enumerate(items_tries):
            if isinstance(gram, tuple):
                texte_gram = " ".join(gram)
            else:
                texte_gram = str(gram)
            tag = "pair" if idx % 2 == 0 else "impair"
            arbre.insert("", tk.END, values=(texte_gram, freq, f"{prob:.4f}"), tags=(tag,))

    def action_exporter_tout(self):
        if not hasattr(self, "articles") or not self.articles:
            messagebox.showwarning("Attention", 
                                 "Aucun article à exporter. Chargez d'abord un numéro.")
            return
        dossier = filedialog.askdirectory(initialdir=OUTPUT_DIR, 
                                         title="Choisir le dossier de destination")
        if not dossier:
            return
        threading.Thread(target=self._thread_export, args=(dossier,), daemon=True).start()

    def _thread_export(self, dossier):
        try:
            self.var_statut.set("⏳ Export en cours...")
            choix_stem = self.var_stemmer.get()
            for idx, (titre, lien) in enumerate(self.articles, start=1):
                t, resume = extract_title_and_abstract(lien)
                tokens = tokenize_and_normalize(resume, choix_stem)
                uni = compute_ngrams(tokens, 1)
                bi = compute_ngrams(tokens, 2)
                tri = compute_ngrams(tokens, 3)
                
                nom_fichier = f"Article_{idx:03d}"
                with open(os.path.join(dossier, f"{nom_fichier}_metadonnees.txt"), 
                         "w", encoding="utf-8") as f:
                    f.write(f"Titre: {t}\nURL: {lien}\n\nRésumé:\n{resume}\n")
                
                def ecrire_ngrams(nom_fich, dict_ngrams):
                    with open(os.path.join(dossier, nom_fich), "w", encoding="utf-8") as f:
                        items_tries = sorted(dict_ngrams.items(), 
                                           key=lambda x: x[1][0], reverse=True)
                        for gram, (freq, prob) in items_tries:
                            texte = " ".join(gram) if isinstance(gram, tuple) else str(gram)
                            f.write(f"{texte}\t{freq}\t{prob:.6f}\n")
                
                ecrire_ngrams(f"{nom_fichier}_unigrammes.txt", uni)
                ecrire_ngrams(f"{nom_fichier}_bigrammes.txt", bi)
                ecrire_ngrams(f"{nom_fichier}_trigrammes.txt", tri)
                time.sleep(DELAY)
            
            self.var_statut.set(f"✓ Export terminé : {len(self.articles)} articles")
            messagebox.showinfo("Export réussi", 
                              f"Les fichiers ont été enregistrés dans :\n{dossier}")
        except Exception as e:
            self.var_statut.set("✗ Erreur pendant l'export")
            messagebox.showerror("Erreur d'export", str(e))

# ----------------- Lancement de l'application -----------------
if __name__ == "__main__":
    app = ExtracteurNgrams()
    app.mainloop()