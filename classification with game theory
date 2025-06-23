import os
import numpy as np
from collections import Counter
from tqdm import tqdm
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier
from sklearn.cluster import KMeans
from joblib import Parallel, delayed
import pandas as pd
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

class OptimizedTextComplexityAnalyzer:
    def __init__(self, q_p=0.75, n_jobs=4):
        self.q_p = q_p
        self.n_jobs = n_jobs
        
    def preprocess_text(self, text):
        words = [w.strip('.,!?;:"\'()[]') for w in text.split() if w.strip()]
        return [len(w) for w in words if len(w) > 0]

    def create_voting_game(self, word_lengths):
        length_counts = Counter(word_lengths)
        max_len = max(length_counts.keys()) if length_counts else 0
        weights = [length_counts.get(i, 0) for i in range(1, max_len + 1)]
        return weights, sum(length_counts.values()) * self.q_p

    def compute_banzaf_optimized(self, weights, quota):
        n = len(weights)
        if n == 0:
            return np.array([])
            
        dp = [0] * (int(quota) + 1)
        dp[0] = 1
        switch_counts = np.zeros(n)

        for i, w in enumerate(weights):
            w_int = int(w)
            if w_int == 0:
                continue
            for s in range(int(quota), w_int - 1, -1):
                if dp[s - w_int] > 0:
                    if s >= quota:
                        switch_counts[i] += dp[s - w_int]
                    dp[s] += dp[s - w_int]

        total = max(1, sum(switch_counts))
        return switch_counts / total

    def analyze_single_text(self, text):
        word_lengths = self.preprocess_text(text)
        weights, quota = self.create_voting_game(word_lengths)
        
        
        game_features = self.compute_banzaf_optimized(weights, quota)
        game_features = np.pad(game_features, (0, max(0, 17 - len(game_features))))
        
        syntax_features = np.array([
            text.count(' что ') + text.count(' который '),  
            text.count('ся ') + text.count('сь '),          
            len([w for w in text.split() if w.endswith('ть')])  
        ])
        
        stats = np.array([
            len(word_lengths),                              
            np.mean(word_lengths) if word_lengths else 0, 
            np.std(word_lengths) if word_lengths else 0     
        ])
        
        return np.concatenate([game_features[:17], syntax_features, stats])

    def parallel_analyze(self, texts):
        try:
            results = Parallel(n_jobs=self.n_jobs, prefer="threads")(
                delayed(self.analyze_single_text)(text) 
                for text in texts
            )
            return np.array(results)
        except Exception as e:
            return np.array([self.analyze_single_text(text) for text in texts])

    def cluster_texts(self, texts, n_clusters=5):
        vectors = self.parallel_analyze(texts)
        if len(vectors) == 0:
            return np.array([])
            
        kmeans = KMeans(n_clusters=min(n_clusters, len(vectors)), n_init='auto')
        return kmeans.fit_predict(vectors)

def load_texts_safe(data_path):
    levels = ['A1', 'A2', 'B1', 'B2', 'C1']
    texts, labels = [], []
    
    for level in levels:
        level_path = os.path.join(data_path, level)
        
            
        files = [f for f in os.listdir(level_path) if f.endswith('.txt')]
        
            
        for file in tqdm(files, desc=f'Loading {level}'):
            try:
                with open(os.path.join(level_path, file), 'r', encoding='utf-8') as f:
                    text = f.read().strip()
                    if text:
                        texts.append(text)
                        labels.append(level)
            except Exception as e:
                print(f"Error reading {file}: {e}")
    
    
    return texts, labels

def evaluate_model_safe(analyzer, texts, labels):
    X = analyzer.parallel_analyze(texts)
    y = pd.factorize(labels)[0]
    
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    
    model = CatBoostClassifier(
        iterations=500,
        depth=6,
        learning_rate=0.05,
        thread_count=4,
        verbose=100,
        auto_class_weights='Balanced',
        task_type='CPU'
    )
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, 
                              target_names=pd.Series(labels).cat.categories))
    return model

if __name__ == "__main__":
    DATA_PATH = "/Users/mzsfleee/Downloads/ru_learning_data-plainrussian"  
    
    try:
        texts, labels = load_texts_safe(DATA_PATH)
        labels = pd.Categorical(labels, categories=['A1', 'A2', 'B1', 'B2', 'C1'], ordered=True)
        print(f"\nLoaded {len(texts)} texts with distribution:")
        print(pd.Series(labels).value_counts().sort_index())
        
        analyzer = OptimizedTextComplexityAnalyzer(n_jobs=4)
        
        features = analyzer.parallel_analyze(texts)
        print(f"Feature matrix shape: {features.shape}")
        
        model = evaluate_model_safe(analyzer, texts, labels)
        
        cluster_labels = analyzer.cluster_texts(texts)
        print("\nCluster distribution:")
        print(pd.Series(cluster_labels).value_counts().sort_index())
        
    except Exception as e:
        print(f"\nError: {str(e)}")
