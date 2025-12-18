#  takes words and prints snowball stemmed versions
from nltk.stem.snowball import SnowballStemmer
def stem_words(words, language='english'):
    stemmer = SnowballStemmer(language)
    return [stemmer.stem(word) for word in words]
if __name__ == "__main__":
    words = ["worst", "accuracies", "terms", "complexity","reduce","wolf","particle",
             "colony","whale","swarm","shop","salesman","travelling","np","dispatch",
             "recall","precision","f1","specificity","proposes","presents","introduces",
             "introduce","aims"
             ]
    stemmed_words = stem_words(words)
    print("Original words:", words)
    print("Stemmed words:", stemmed_words)
