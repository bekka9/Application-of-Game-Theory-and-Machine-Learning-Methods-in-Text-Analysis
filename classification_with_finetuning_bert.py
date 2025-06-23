import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer, 
    AutoModel,
    TrainingArguments, 
    Trainer,
    TrainerCallback
)
import numpy as np
import os
import logging
import pandas as pd
import re
from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from collections import Counter
from natasha import Segmenter, MorphVocab, NewsEmbedding, NewsMorphTagger, NewsSyntaxParser, Doc
from razdel import tokenize, sentenize
import matplotlib.pyplot as plt
import seaborn as sns
import multiprocessing

multiprocessing.set_start_method('spawn', force=True)

segmenter = Segmenter()
morph_vocab = MorphVocab()
emb = NewsEmbedding()
morph_tagger = NewsMorphTagger(emb)
syntax_parser = NewsSyntaxParser(emb)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def count_syllables(word):
    vowels = 'аеёиоуыэюя'
    return sum(1 for char in word.lower() if char in vowels) or 1

class TextDifficultyDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length, features=None):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.features = features

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        item = {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }
        
        if self.features is not None:
            item['features'] = torch.tensor(self.features[idx], dtype=torch.float)
            
        return item

class HybridBertModel(torch.nn.Module):
    def __init__(self, model_name, num_labels, feature_dim=0):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.feature_dim = feature_dim
        self.bert_hidden_size = self.bert.config.hidden_size
        
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(self.bert_hidden_size + feature_dim, 256),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(256, num_labels)
        )

    def forward(self, input_ids, attention_mask, features=None, labels=None):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        pooled_output = outputs.last_hidden_state[:, 0, :]
        
        if self.feature_dim > 0 and features is not None:
            combined = torch.cat([pooled_output, features], dim=1)
        else:
            combined = pooled_output
            
        logits = self.classifier(combined)
        
        loss = None
        if labels is not None:
            loss_fct = torch.nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.classifier[-1].out_features), labels.view(-1))
        
        return {'loss': loss, 'logits': logits}

class ProgressCallback(TrainerCallback):
    def __init__(self):
        self.progress = None
        
    def on_train_begin(self, args, state, control, **kwargs):
        self.progress = tqdm(total=state.max_steps, 
                            desc="Обучение модели", 
                            bar_format="{l_bar}{bar:40}{r_bar}", 
                            dynamic_ncols=True)
        
    def on_step_end(self, args, state, control, **kwargs):
        self.progress.update(1)
        if state.log_history:
            self.progress.set_postfix({
                'loss': f"{state.log_history[-1].get('loss', 0):.4f}",
                'epoch': f"{state.epoch:.1f}"
            })
            
    def on_train_end(self, args, state, control, **kwargs):
        self.progress.close()

def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=1)
    return {
        'accuracy': accuracy_score(labels, predictions),
        'macro_f1': f1_score(labels, predictions, average='macro')
    }

def process_text(text):
    try:
        words = [t.text for t in tokenize(text)]
        sentences = [s.text for s in sentenize(text)]
        
        doc = Doc(text)
        doc.segment(segmenter)
        doc.tag_morph(morph_tagger)
        doc.parse_syntax(syntax_parser)

        stats = {
            'n_words': len(words),
            'n_sentences': len(sentences),
            'complex_words': 0,
            'total_syllables': 0,
            'passive_constructions': 0,
            'subordinate_clauses': 0,
            'modal_verbs': 0,
            'conditional_mood': 0
        }

        possible_pos = [
            'NOUN', 'VERB', 'ADJ', 'ADV', 'ADP', 'DET', 'SCONJ',
            'NUM', 'PRON', 'CCONJ', 'PART', 'INTJ', 'PROPN', 
            'SYM', 'PUNCT', 'X', 'AUX'
        ]
        stats.update({f'pos_{pos.lower()}': 0 for pos in possible_pos})

        pos_counts = Counter()
        for token in doc.tokens:
            try:
                token.lemmatize(morph_vocab)
                pos = token.pos
                pos_counts[pos] += 1

                syllables = count_syllables(token.text)
                stats['total_syllables'] += syllables
                
                if syllables >= 4 or len(token.text) > 8:
                    stats['complex_words'] += 1
                
                if token.rel == 'subord':
                    stats['subordinate_clauses'] += 1
                if 'Voice=Pass' in token.feats:
                    stats['passive_constructions'] += 1
                
                if token.lemma in ['мочь', 'хотеть', 'должен']:
                    stats['modal_verbs'] += 1
                if 'Mood=Cnd' in token.feats:
                    stats['conditional_mood'] += 1

            except Exception as e:
                logging.warning(f"Ошибка обработки токена {token.text}: {str(e)}")
                continue

        for pos, count in pos_counts.items():
            key = f'pos_{pos.lower()}'
            if key in stats:
                stats[key] = count
            else:
                logging.warning(f"Неизвестная часть речи: {pos}")

        stats.update({
            'pos_diversity': len(pos_counts)/stats['n_words'] if stats['n_words'] >0 else 0,
            'lex_diversity': len({t.lemma for t in doc.tokens})/stats['n_words'] if stats['n_words'] >0 else 0,
            'complex_word_ratio': stats['complex_words']/stats['n_words'] if stats['n_words'] >0 else 0,
            'avg_syll_per_word': stats['total_syllables']/stats['n_words'] if stats['n_words'] >0 else 0,
            'avg_word_len': np.mean([len(w) for w in words]) if words else 0,
            'avg_sent_len': stats['n_words']/len(sentences) if sentences else 0
        })

        return stats
    
    except Exception as e:
        logging.error(f"Критическая ошибка обработки текста: {str(e)}")
        return None

def load_data(data_path):
    levels = ['A1', 'A2', 'B1', 'B2', 'C1']
    data = []
    
    with tqdm(total=len(levels), desc="Обработка уровней") as level_pbar:
        for level in levels:
            level_dir = os.path.join(data_path, level)
            if not os.path.exists(level_dir):
                raise FileNotFoundError(f"Директория {level_dir} не найдена")
            
            files = [f for f in os.listdir(level_dir) if f.endswith('.txt')]
            
            with tqdm(total=len(files), desc=f"Уровень {level}", leave=False) as file_pbar:
                for file in files:
                    file_path = os.path.join(level_dir, file)
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            text = f.read(5000)
                            data.append({'text': text, 'label': level})
                    except Exception as e:
                        logging.warning(f"Ошибка чтения {file_path}: {str(e)}")
                    finally:
                        file_pbar.update(1)
            
            level_pbar.update(1)
   
    return pd.DataFrame(data)

def preprocess_data(df):
    feature_columns = [
        'n_words', 'n_sentences', 'complex_words', 'total_syllables',
        'passive_constructions', 'subordinate_clauses', 'modal_verbs',
        'conditional_mood', 'pos_noun', 'pos_verb', 'pos_adj', 'pos_adv',
        'pos_diversity', 'lex_diversity', 'complex_word_ratio',
        'avg_syll_per_word', 'avg_word_len', 'avg_sent_len'
    ]
    
    features = []
    with tqdm(total=len(df), desc="Извлечение признаков") as pbar:
        for text in df['text']:
            stats = process_text(text)
            features.append([stats.get(col, 0) for col in feature_columns] if stats else [0]*len(feature_columns))
            pbar.update(1)
    
    with tqdm(total=3, desc="Предобработка признаков") as pbar:
        imputer = SimpleImputer(strategy='median')
        features = imputer.fit_transform(features)
        pbar.update(1)
        
        scaler = StandardScaler()
        features = scaler.fit_transform(features)
        pbar.update(1)
        
        df = df[(df['text'].str.split().str.len() > 5)].copy()
        df['label'] = LabelEncoder().fit_transform(df['label'])
        pbar.update(1)
    
    return df, features

def train_transformer_model(df, features, target_labels):
    tokenizer = AutoTokenizer.from_pretrained("cointegrated/rubert-tiny2")
    
    dataset = TextDifficultyDataset(
        texts=df['text'].tolist(),
        labels=df['label'].values,
        tokenizer=tokenizer,
        max_length=256,
        features=features
    )
    
    train_idx, val_idx = train_test_split(
        np.arange(len(dataset)),
        test_size=0.2,
        stratify=df['label'],
        random_state=42
    )
    
    model = HybridBertModel(
        model_name="cointegrated/rubert-tiny2",
        num_labels=len(target_labels),
        feature_dim=features.shape[1]
    )
    
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=5,
        per_device_train_batch_size=32,
        gradient_accumulation_steps=2,
        learning_rate=2e-5,
        weight_decay=0.01,
        warmup_ratio=0.1,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        logging_dir='./logs',
        logging_steps=50,
        fp16=torch.cuda.is_available(),
        dataloader_num_workers=0,
        report_to="none",
        disable_tqdm=True
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=torch.utils.data.Subset(dataset, train_idx),
        eval_dataset=torch.utils.data.Subset(dataset, val_idx),
        compute_metrics=compute_metrics,
        callbacks=[ProgressCallback()]
    )
    
    trainer.train()
    return trainer, tokenizer

def visualize_training(trainer, y_true, y_pred, classes):
    plt.figure(figsize=(18, 6))
    
    plt.subplot(1, 3, 1)
    train_loss = [x['loss'] for x in trainer.state.log_history if 'loss' in x]
    eval_loss = [x['eval_loss'] for x in trainer.state.log_history if 'eval_loss' in x]
    plt.plot(train_loss, label='Training Loss')
    plt.plot(eval_loss, label='Validation Loss')
    plt.title('Динамика потерь')
    plt.xlabel('Шаг')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 3, 2)
    eval_acc = [x['eval_accuracy'] for x in trainer.state.log_history if 'eval_accuracy' in x]
    plt.plot(eval_acc, label='Validation Accuracy')
    plt.title('Динамика точности')
    plt.xlabel('Шаг')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 3, 3)
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, yticklabels=classes)
    plt.title('Матрица ошибок')
    plt.xlabel('Предсказанный класс')
    plt.ylabel('Истинный класс')
    
    plt.tight_layout()
    plt.show()

def main():
    try:
        data_path = "/Users/mzsfleee/Downloads/ru_learning_data-plainrussian"
        df = load_data(data_path)
        
      
        df_clean = df[df['text'].notna() & (df['text'] != '')].copy()
        
        df_clean['text'] = df_clean['text'].str.lower().str.replace(r"[^а-яё\s]", "", regex=True)
        df_preprocessed, features = preprocess_data(df_clean)
        
        target_labels = ['A1', 'A2', 'B1', 'B2', 'C1']
        
        trainer, tokenizer = train_transformer_model(df_preprocessed, features, target_labels)
        
        trainer.save_model("text_difficulty_model")
        tokenizer.save_pretrained("text_difficulty_tokenizer")
        
        predictions = trainer.predict(trainer.eval_dataset)
        y_pred = np.argmax(predictions.predictions, axis=1)
        y_true = predictions.label_ids
        visualize_training(trainer, y_true, y_pred, target_labels)
        
    except Exception as e:
        logging.error(f"Критическая ошибка: {str(e)}")
        raise
    finally:
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
