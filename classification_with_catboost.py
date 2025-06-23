import os
import re
import zipfile
import pandas as pd
import numpy as np
import logging
from natasha import (
    Segmenter,
    MorphVocab,
    NewsEmbedding,
    NewsMorphTagger,
    NewsSyntaxParser,
    Doc
)
from razdel import tokenize, sentenize
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from collections import Counter
from time import time
import matplotlib.pyplot as plt
import seaborn as sns
import optuna
from optuna.samplers import TPESampler
from optuna.integration import CatBoostPruningCallback
import torch
from transformers import AutoTokenizer, AutoModel
from functools import lru_cache

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
device = 'cuda' if torch.cuda.is_available() else 'cpu'

segmenter = Segmenter()
morph_vocab = MorphVocab()
emb = NewsEmbedding()
morph_tagger = NewsMorphTagger(emb)
syntax_parser = NewsSyntaxParser(emb)

tokenizer = AutoTokenizer.from_pretrained("cointegrated/rubert-tiny2")
model = AutoModel.from_pretrained("cointegrated/rubert-tiny2").to(device)

def split_long_text(text, max_length=1000):
    return [text[i:i+max_length] for i in range(0, len(text), max_length)]

def count_syllables(word):
    vowels = 'аеёиоуыэюя'
    return sum(1 for char in word.lower() if char in vowels) or 1

@lru_cache(maxsize=1000)
def cached_semantic_features(text):
    try:
        inputs = tokenizer(text, return_tensors='pt', max_length=512, truncation=True).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy().squeeze()
        return {
            'semantic_density': np.std(embeddings),
            'cls_embedding': embeddings[0] if embeddings.ndim > 0 else embeddings
        }
    except Exception as e:
        logging.error(f"Semantic error: {str(e)}")
        return {}

def get_stylistic_features(text):
    idioms = [
        'бить баклуши', 'водить за нос', 'дело в шляпе', 
        'кот наплакал', 'медвежья услуга'
    ]
    
    metaphors = 0
    terms = 0
    idiom_count = 0
    
    for sentence in sentenize(text):
        sent = sentence.text.lower()
        idiom_count += sum(1 for idiom in idioms if idiom in sent)
        
        if any(word in sent for word in ['как', 'словно', 'будто']):
            metaphors += 1
            
        if re.search(r'«.*?»|[A-ZА-Я]{3,}', sent):
            terms += 1
    
    return {
        'metaphors': metaphors,
        'terms': terms,
        'idioms': idiom_count
    }

def get_coherence_features(doc):
    coref_chains = []
    current_chain = []
    previous_lemmas = set()
    
    for token in doc.tokens:
        token.lemmatize(morph_vocab)
        if token.rel == 'nsubj':
            if current_chain:
                coref_chains.append(current_chain)
                current_chain = []
            current_chain.append(token.lemma)
        elif token.rel in ['obj', 'amod'] and current_chain:
            current_chain.append(token.lemma)
        
        previous_lemmas.add(token.lemma)
    
    return {
        'coref_chains': len(coref_chains),
        'coref_chain_avg_len': np.mean([len(c) for c in coref_chains]) if coref_chains else 0,
        'lexical_repetitions': sum(1 for t in doc.tokens if t.lemma in previous_lemmas),
        'pronoun_ratio': sum(1 for t in doc.tokens if t.pos == 'PRON') / len(doc.tokens) if doc.tokens else 0
    }

def get_text_stats(text):
    words = [t.text for t in tokenize(text)]
    sentences = [s.text for s in sentenize(text)]
    
    doc = Doc(text)
    doc.segment(segmenter)
    doc.tag_morph(morph_tagger)
    
    lemmas = []
    complex_words = 0
    total_syllables = 0
    passive_constructions = 0
    subordinate_clauses = 0
    modal_verbs = 0
    conditional_mood = 0
    
    for token in doc.tokens:
        token.lemmatize(morph_vocab)
        lemmas.append(token.lemma)
        
        syllables = count_syllables(token.text)
        total_syllables += syllables
        complex_criteria = (
            syllables >= 4 or
            len(token.text) > 8 or
            token.text.endswith(('ость', 'ение')) or
            'ся' in token.text
        )
        if complex_criteria:
            complex_words += 1
        
        if token.rel == 'subord':
            subordinate_clauses += 1
        if 'Voice=Pass' in token.feats:
            passive_constructions += 1
        
        if token.lemma in ['мочь', 'хотеть', 'должен']:
            modal_verbs += 1
        if 'Mood=Cnd' in token.feats:
            conditional_mood += 1
    
    unique_lemmas = len(set(lemmas))
    avg_syll_per_word = total_syllables / len(words) if words else 0
    
    return {
        'n_words': len(words),
        'n_sentences': len(sentences),
        'avg_word_len': np.mean([len(w) for w in words]) if words else 0,
        'avg_sent_len': len(words)/len(sentences) if sentences else 0,
        'lex_diversity': len(set(lemmas))/len(words) if words else 0,
        'complex_word_ratio': complex_words/len(words) if words else 0,
        'total_syllables': total_syllables,
        'subordinate_clauses': subordinate_clauses,
        'passive_constructions': passive_constructions,
        'modal_verbs': modal_verbs,
        'conditional_mood': conditional_mood,
        'avg_syll_per_word': avg_syll_per_word
    }

def extract_features(text):
    try:
        stats = get_text_stats(text)
        stylistic = get_stylistic_features(text)
        
        doc = Doc(text)
        doc.segment(segmenter)
        doc.tag_morph(morph_tagger)
        doc.parse_syntax(syntax_parser)
        
        coherence = get_coherence_features(doc)
        semantic = cached_semantic_features(text[:512])  # Обрезаем для BERT
        
        pos_counts = Counter()
        for token in doc.tokens:
            pos_counts[token.pos] += 1
        
        return {
            **stats,
            **stylistic,
            **coherence,
            **semantic,
            'pos_noun': pos_counts['NOUN'],
            'pos_verb': pos_counts['VERB'],
            'pos_adj': pos_counts['ADJ'],
            'pos_adv': pos_counts['ADV'],
            'pos_diversity': len(pos_counts)/stats['n_words'] if stats['n_words'] > 0 else 0
        }
        
    except Exception as e:
        logging.error(f"Feature extraction error: {str(e)}")
        return None

def process_data(data_path, max_text_length=1000):
    data = []
    labels = []
    levels = ['A1', 'A2', 'B1', 'B2', 'C1']
    
    for level in levels:
        level_dir = os.path.join(data_path, level)
            
        filenames = [f for f in os.listdir(level_dir) if f.endswith('.txt')]
        for filename in tqdm(filenames, desc=f"Обработка {level}"):
            file_path = os.path.join(level_dir, filename)
            
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read().strip()
                    if not text:
                        continue
                    
                    text_chunks = split_long_text(text, max_text_length)
                    for chunk in text_chunks:
                        features = extract_features(chunk)
                        if features:
                            data.append(features)
                            labels.append(level)
                            
            except Exception as e:
                logging.error(f"Ошибка файла {filename}: {str(e)}")
    
    return pd.DataFrame(data), labels

def visualize_features(df):
    features = [
        'pos_diversity', 'n_words', 'n_sentences', 'avg_word_len', 'avg_sent_len',
        'lex_diversity', 'complex_word_ratio', 'total_syllables', 'avg_syll_per_word',
        'metaphors', 'terms', 'idioms', 'coref_chains', 'coref_chain_avg_len',
        'lexical_repetitions', 'pronoun_ratio', 'semantic_density'
    ]
    
    plt.figure(figsize=(18, 12))
    sns.pairplot(df[features[:8] + ['label']], hue='label', palette='viridis', height=3)
    plt.suptitle('Попарные отношения базовых признаков', y=1.02)
    plt.tight_layout()
    plt.show()
    
    plt.figure(figsize=(16, 12))
    corr_matrix = df[features + ['label']].apply(lambda x: pd.factorize(x)[0]).corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", annot_kws={"size": 8})
    plt.title('Матрица корреляций признаков')
    plt.tight_layout()
    plt.show()

def objective(trial, X_train, y_train, X_val, y_val):
    params = {
        'iterations': trial.suggest_int('iterations', 800, 1500),
        'depth': trial.suggest_int('depth', 8, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10),
        'random_strength': trial.suggest_float('random_strength', 0.1, 10),
        'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 1.0),
        'border_count': trial.suggest_int('border_count', 32, 255),
        'verbose': False,
        'auto_class_weights': 'Balanced',
        'early_stopping_rounds': 20,
        'task_type': 'GPU' if torch.cuda.is_available() else 'CPU',
        'eval_metric': 'Accuracy',
        'loss_function': 'MultiClass'
    }
    
    model = CatBoostClassifier(**params)
    pruning_callback = CatBoostPruningCallback(trial, 'MultiClass')
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[pruning_callback],
        verbose=0
    )
    
    y_pred = model.predict(X_val)
    return accuracy_score(y_val, y_pred)

def optimize_hyperparameters(X_train, y_train, n_trials=50):
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, stratify=y_train, random_state=42
    )
    
    study = optuna.create_study(
        direction='maximize',
        sampler=TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=10)
    )
    
    study.optimize(
        lambda trial: objective(trial, X_train, y_train, X_val, y_val),
        n_trials=n_trials,
        timeout=3600,
        show_progress_bar=True
    )
    
    best_params = study.best_params.copy()
    best_params.update({
        'task_type': 'GPU' if torch.cuda.is_available() else 'CPU',
        'verbose': 100,
        'eval_metric': 'Accuracy',
        'loss_function': 'MultiClass'
    })
    
    best_model = CatBoostClassifier(**best_params)
    best_model.fit(X_train, y_train)
    
    return best_model, study

def main():
    try:
        start_time = time()
        data_path = '/Users/mzsfleee/Downloads/ru_learning_data-plainrussian'
        
        logging.info("Начало обработки данных...")
        df, labels = process_data(data_path)
        df['label'] = labels
        
        df = df.drop_duplicates()
        imputer = SimpleImputer(strategy='median')
        df_imputed = pd.DataFrame(
            imputer.fit_transform(df.drop('label', axis=1)),
            columns=df.columns[:-1]
        )
        df_imputed['label'] = df['label'].values
        
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(df_imputed.drop('label', axis=1))
        df_scaled = pd.DataFrame(scaled_features, columns=df.columns[:-1])
        df_scaled['label'] = df_imputed['label'].values
        
        X_train, X_test, y_train, y_test = train_test_split(
            df_scaled.drop('label', axis=1),
            df_scaled['label'],
            test_size=0.2,
            stratify=df_scaled['label'],
            random_state=42
        )
        
        best_model, study = optimize_hyperparameters(X_train, y_train)
        y_pred = best_model.predict(X_test)
        
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        visualize_features(df_imputed)
        
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
