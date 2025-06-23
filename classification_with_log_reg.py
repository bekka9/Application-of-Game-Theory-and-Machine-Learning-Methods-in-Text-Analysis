import os
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
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tqdm import tqdm
from collections import Counter
from time import time
import matplotlib.pyplot as plt
import seaborn as sns
import optuna
from optuna.samplers import TPESampler
import torch

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

segmenter = Segmenter()
morph_vocab = MorphVocab()
emb = NewsEmbedding()
morph_tagger = NewsMorphTagger(emb)
syntax_parser = NewsSyntaxParser(emb)

def split_long_text(text, max_length=1000):
    return [text[i:i+max_length] for i in range(0, len(text), max_length)]

def count_syllables(word):
    vowels = 'аеёиоуыэюя'
    return sum(1 for char in word.lower() if char in vowels) or 1

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
        
        doc = Doc(text)
        doc.segment(segmenter)
        doc.tag_morph(morph_tagger)
        doc.parse_syntax(syntax_parser)
        
        pos_counts = Counter()
        for token in doc.tokens:
            pos_counts[token.pos] += 1
        
        return {
            **stats,
            'pos_noun': pos_counts['NOUN'],
            'pos_verb': pos_counts['VERB'],
            'pos_adj': pos_counts['ADJ'],
            'pos_adv': pos_counts['ADV'],
            'pos_diversity': len(pos_counts)/stats['n_words'] if stats['n_words'] > 0 else 0
        }
        
    except Exception as e:
        logging.error(f"{str(e)}")
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
        'lex_diversity', 'complex_word_ratio', 'total_syllables', 'avg_syll_per_word'
    ]
    
    num_features = len(features)
    features_per_figure = 6
    
    for i in range(0, num_features, features_per_figure):
        current_features = features[i:i+features_per_figure]
        rows = min(2, len(current_features))
        cols = min(3, len(current_features))
        
        plt.figure(figsize=(18, 12))
        for j, feature in enumerate(current_features, 1):
            plt.subplot(rows, cols, j)
            sns.histplot(df[feature], kde=True, bins=20)
            plt.title(f'Распределение {feature}')
        plt.tight_layout()
        plt.show()
        
        plt.figure(figsize=(18, 12))
        for j, feature in enumerate(current_features, 1):
            plt.subplot(rows, cols, j)
            sns.boxplot(x='label', y=feature, data=df)
            plt.title(f'{feature} по уровням CEFR')
            plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
    
    plt.figure(figsize=(12, 10))
    corr_matrix = df[features + ['label']].apply(lambda x: pd.factorize(x)[0]).corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Матрица корреляций признаков')
    plt.tight_layout()
    plt.show()
    
    sns.pairplot(df[features[:6] + ['label']], hue='label', palette='viridis', height=3)
    plt.suptitle('Попарные отношения признаков (1-6)', y=1.02)
    plt.tight_layout()
    plt.show()
    
    sns.pairplot(df[features[6:] + ['label']], hue='label', palette='viridis', height=3)
    plt.suptitle('Попарные отношения признаков (7-13)', y=1.02)
    plt.tight_layout()
    plt.show()

def objective(trial, X_train, y_train, X_val, y_val):
    params = {
        'C': trial.suggest_float('C', 0.01, 10, log=True),
        'penalty': trial.suggest_categorical('penalty', ['l1', 'l2', 'elasticnet']),
        'solver': trial.suggest_categorical('solver', ['saga']),  
        'l1_ratio': trial.suggest_float('l1_ratio', 0, 1) if trial.params['penalty'] == 'elasticnet' else None,
        'max_iter': trial.suggest_int('max_iter', 100, 1000),
        'class_weight': trial.suggest_categorical('class_weight', [None, 'balanced']),
        'multi_class': 'multinomial'
    }
    
    if params['penalty'] != 'elasticnet':
        del params['l1_ratio']
    
    model = LogisticRegression(**params)
    
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)
    
    return accuracy

def optimize_hyperparameters(X_train, y_train, n_trials=50):
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, stratify=y_train, random_state=42
    )
    
    study = optuna.create_study(
        direction='maximize',
        sampler=TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=10)
    )
    
    func = lambda trial: objective(trial, X_train, y_train, X_val, y_val)
    
    study.optimize(
        func,
        n_trials=n_trials,
        timeout=3600, 
        show_progress_bar=True
    )
    
    print("\nЛучшие параметры:")
    for key, value in study.best_params.items():
        print(f"{key}: {value}")
    print(f"Лучшая точность: {study.best_value:.4f}")
    
    best_params = study.best_params.copy()
    best_model = LogisticRegression(**best_params)
    best_model.fit(X_train, y_train)
    
    return best_model, study

def main():
    try:
        
        start_time = time()
        data_path = '/Users/mzsfleee/Downloads/ru_learning_data-plainrussian'
        
        df, labels = process_data(data_path, max_text_length=1000)
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
        
        visualize_features(df_imputed)
        best_model, study = optimize_hyperparameters(X_train, y_train, n_trials=50)
        
        y_pred = best_model.predict(X_test)
        
        print("\n" + "="*50)
        print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        print("="*50)
        
        feature_importance = pd.DataFrame({
            'feature': X_train.columns,
            'coefficient': best_model.coef_[0]  
        }).sort_values('coefficient', key=abs, ascending=False)
        
        plt.figure(figsize=(12, 8))
        sns.barplot(x='coefficient', y='feature', data=feature_importance.head(20))
        plt.title('Top 20 важных признаков')
        plt.tight_layout()
        plt.show()
        
        report_name = "classification_report_logreg.txt"
        with open(report_name, 'w') as f:
            f.write(f"Best parameters: {best_model.get_params()}\n")
            f.write(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}\n\n")
            f.write("Classification Report:\n")
            f.write(classification_report(y_test, y_pred))
        
        
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
