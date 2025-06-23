import os
import logging
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from catboost import CatBoostClassifier
from natasha import Segmenter, MorphVocab, NewsEmbedding, NewsMorphTagger

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class TextDataLoader:
    def __init__(self, base_path):
        self.base_path = base_path
        self.levels = ['A1', 'A2', 'B1', 'B2', 'C1']
        self.segmenter = Segmenter()
        self.morph_vocab = MorphVocab()
        self.emb = NewsEmbedding()
        self.morph_tagger = NewsMorphTagger(self.emb)
    
    def load_data(self):
        data = []
        labels = []
        
        for level in self.levels:
            level_path = os.path.join(self.base_path, level)
            if not os.path.exists(level_path):
                continue
                
            files = [f for f in os.listdir(level_path) if f.endswith('.txt')]
            for file in tqdm(files, desc=f'Loading {level}'):
                try:
                    with open(os.path.join(level_path, file), 'r', encoding='utf-8') as f:
                        text = f.read()
                        features = self.extract_features(text)
                        data.append(features)
                        labels.append(level)
                except Exception as e:
                    logging.error(f"Error loading {file}: {str(e)}")
        
        return pd.DataFrame(data), labels
    
    def extract_features(self, text):
        doc = Doc(text)
        doc.segment(self.segmenter)
        doc.tag_morph(self.morph_tagger)
        
        words = [token.text for token in doc.tokens]
        sentences = list(doc.sents)
        
        features = {
            'num_words': len(words),
            'num_sentences': len(sentences),
            'avg_word_length': np.mean([len(w) for w in words]) if words else 0,
            'passive_voice': sum(1 for t in doc.tokens if 'Voice=Pass' in t.feats),
            'noun_ratio': sum(1 for t in doc.tokens if 'NOUN' in t.pos) / len(words) if words else 0,
        }
        
        game_features = self.game_theory_features(words)
        features.update(game_features)
        
        return features
    
    def game_theory_features(self, words):
        word_lengths = [len(w) for w in words]
        length_counts = pd.Series(word_lengths).value_counts().to_dict()
        
        total = sum(length_counts.values())
        weights = {k: v/total for k, v in length_counts.items()}
        
        return {
            'shannon_entropy': -sum(p * np.log2(p) for p in weights.values()),
            'power_imbalance': max(weights.values()) - min(weights.values()),
            'effective_words': len([v for v in weights.values() if v > 0.1])
        }

class TextClassifier:
    def __init__(self):
        self.model = CatBoostClassifier(
            iterations=1000,
            depth=8,
            learning_rate=0.1,
            verbose=100
        )
    
    def train(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )
        
        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_test)
        
        print(classification_report(y_test, y_pred))
    
    def save_model(self, path):
        self.model.save_model(path)
        logging.info(f"Model saved to {path}")

if __name__ == "__main__":
    DATA_PATH = "/Users/mzsfleee/Downloads/ru_learning_data-plainrussian"
    
    loader = TextDataLoader(DATA_PATH)
    df, labels = loader.load_data()
    
    label_map = {level: idx for idx, level in enumerate(loader.levels)}
    y = pd.Series(labels).map(label_map)
    
    classifier = TextClassifier()
    classifier.train(df, y)
    
    classifier.save_model("text_classifier.cbm")
